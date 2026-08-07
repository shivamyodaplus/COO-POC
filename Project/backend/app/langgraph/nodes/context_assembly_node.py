"""
ContextAssemblyNode — Reduce phase, step 3 of the Map-Reduce RAG pipeline.

Condenses the retrieved PACD chunks (typically 5-18 chunks from up to 5 PACD
documents) into a structured XML context string ready for the LLM critic.

Structure of ``assembled_context``
------------------------------------
The output is a self-contained XML block consumed by ``llm_critic_node``:

    <target_coo>
      [COO header fields as key: value pairs]
      [COO line items as a markdown table]
    </target_coo>

    <filtered_reference_context>
      <doc id="commercial_invoice" doc_type="commercial_invoice" relevant_pages="[4,5,6]">
        === header ===
        [key: value pairs]

        === table (items 1-20) ===
        | # | HS Code | Description | ... |
        |---|---------|-------------|-----|
        | 1 | 7318.15 | Bolts M8    | ... |

        === footer ===
        [key: value pairs]
      </doc>
      <doc id="packing_list" ...>
        ...
      </doc>
      <context_note>
        Retrieved N chunks from M PACD documents. Some sections may be missing.
      </context_note>
    </filtered_reference_context>

Grouping logic
--------------
Chunks are grouped by ``(doc_type, document_id)`` and ordered by ``chunk_index``
within each group.  Multiple documents of the same type (e.g. two invoices)
each get their own ``<doc>`` block distinguished by a numeric suffix.

Page numbers from ``page_numbers`` field (JSON list) are merged across chunks
of the same logical document to produce the ``relevant_pages`` attribute.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

# Maximum character budget for the assembled context string.
# Keeps the final LLM prompt under ~15-20K tokens even with many chunks.
_MAX_CONTEXT_CHARS = 40_000


def _coo_to_text(coo_data: dict[str, Any]) -> str:
    """Format COO extracted data as markdown."""
    lines: list[str] = []

    header_fields: dict[str, str] = coo_data.get("header_fields", {})
    if header_fields:
        lines.append("=== COO Header ===")
        for k, v in header_fields.items():
            lines.append(f"{k}: {v}")

    line_items: list[dict] = coo_data.get("line_items", [])
    if line_items:
        lines.append(f"\n=== COO Line Items ({len(line_items)} items) ===")
        lines.append("| # | HS Code | Description | Qty | Unit | Value | Weight | Origin |")
        lines.append("|---|---------|-------------|-----|------|-------|--------|--------|")
        for item in line_items:
            lines.append(
                f"| {item.get('item_number') or ''} "
                f"| {item.get('hs_code') or ''} "
                f"| {item.get('description') or ''} "
                f"| {item.get('quantity') or ''} "
                f"| {item.get('unit') or ''} "
                f"| {item.get('value') or ''} "
                f"| {item.get('weight') or ''} "
                f"| {item.get('origin_country') or ''} |"
            )

    return "\n".join(lines)


def _merge_page_numbers(page_numbers_list: list[str]) -> list[int]:
    """Merge JSON page-number arrays from multiple chunks."""
    merged: set[int] = set()
    for raw in page_numbers_list:
        try:
            pages = json.loads(raw)
            if isinstance(pages, list):
                merged.update(int(p) for p in pages)
        except Exception:
            pass
    return sorted(merged)


def context_assembly_node(state: GraphState) -> GraphState:
    """LangGraph node: assemble retrieved chunks into a structured XML context.

    Reads
    -----
    state["retrieved_chunks"]   — list of chunk dicts from rag_retrieval_node
    state["coo_extracted_data"] — COO structured extraction

    Writes
    ------
    state["assembled_context"]  — XML string for llm_critic_node
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("context_assembly_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    retrieved_chunks: list[dict[str, Any]] = state.get("retrieved_chunks") or []
    coo_data: dict[str, Any] = state.get("coo_extracted_data") or {}

    # ── Build <target_coo> block ──────────────────────────────────────────
    coo_text = _coo_to_text(coo_data)
    parts: list[str] = [
        "<target_coo>",
        coo_text,
        "</target_coo>",
        "",
        "<filtered_reference_context>",
    ]

    if not retrieved_chunks:
        parts.append(
            "  <context_note>No PACD chunks retrieved. "
            "Ensure PACD documents have been uploaded for this transaction.</context_note>"
        )
        parts.append("</filtered_reference_context>")
        assembled = "\n".join(parts)
        return {
            **state,
            "assembled_context": assembled,
            "errors": errors,
            "current_step": "context_assembly_node",
        }  # type: ignore[return-value]

    # ── Group chunks by (doc_type, document_id) ────────────────────────────
    # Key: (doc_type, document_id)  Value: list of chunks sorted by chunk_index
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in retrieved_chunks:
        key = (chunk.get("doc_type", "unknown"), chunk.get("document_id", ""))
        groups[key].append(chunk)

    # Sort each group by chunk_index
    for key in groups:
        groups[key].sort(key=lambda c: c.get("chunk_index", 0))

    # Build unique doc-type label (add suffix if same doc_type appears in multiple logical docs)
    doc_type_counts: dict[str, int] = defaultdict(int)
    for (doc_type, _) in groups:
        doc_type_counts[doc_type] += 1
    doc_type_seen: dict[str, int] = defaultdict(int)

    total_chars = sum(len(p) for p in parts)
    doc_count = 0

    for (doc_type, document_id), chunks in groups.items():
        doc_type_seen[doc_type] += 1
        count = doc_type_counts[doc_type]
        label = doc_type if count == 1 else f"{doc_type}_{doc_type_seen[doc_type]}"

        # Merge page numbers from all chunks in this group
        all_page_nums = _merge_page_numbers([c.get("page_numbers", "[]") for c in chunks])
        pages_attr = json.dumps(all_page_nums)

        doc_parts: list[str] = [
            f'  <doc id="{label}" doc_type="{doc_type}" relevant_pages="{pages_attr}">',
        ]

        # Emit each chunk's text under a section label
        for chunk in chunks:
            ctype = chunk.get("chunk_type", "")
            ctext = chunk.get("chunk_text", "").strip()
            if not ctext:
                continue
            section_label = {
                "header": "=== header ===",
                "table":  "=== table ===",
                "footer": "=== footer ===",
            }.get(ctype, f"=== {ctype} ===")
            doc_parts.append(f"    {section_label}")
            for line in ctext.splitlines():
                doc_parts.append(f"    {line}")
            doc_parts.append("")

        doc_parts.append("  </doc>")

        doc_text = "\n".join(doc_parts)

        # Enforce global character budget
        if total_chars + len(doc_text) > _MAX_CONTEXT_CHARS:
            logger.warning(
                "context_assembly_node: context budget exhausted at doc '%s' "
                "(%d chars used / %d limit) — truncating",
                label, total_chars, _MAX_CONTEXT_CHARS,
            )
            parts.append(
                f"  <context_note>Context truncated: remaining documents omitted "
                f"(budget {_MAX_CONTEXT_CHARS} chars). {len(groups) - doc_count} docs excluded.</context_note>"
            )
            break

        parts.extend(doc_parts)
        total_chars += len(doc_text)
        doc_count += 1

    parts.append(
        f"  <context_note>Retrieved {len(retrieved_chunks)} chunks from "
        f"{doc_count} PACD document section(s).</context_note>"
    )
    parts.append("</filtered_reference_context>")

    assembled = "\n".join(parts)
    logger.info(
        "context_assembly_node: assembled context = %d chars across %d doc groups",
        len(assembled), doc_count,
    )

    return {
        **state,
        "assembled_context": assembled,
        "errors": errors,
        "current_step": "context_assembly_node",
    }  # type: ignore[return-value]
