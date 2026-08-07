"""
ReportConsolidationNode — merges results from the three parallel verification
nodes (header / content / footer) into a unified discrepancy table and
generates the final verification report narrative via Text LLM.

This node replaces both cross_reference_node and report_generation_node from
the original pipeline.

Reads
-----
state["header_verification_results"]   — list of HeaderFieldVerdict dicts
state["content_verification_results"]  — list of ItemVerificationEntry dicts
state["footer_verification_results"]   — list of FooterFieldVerdict dicts
state["transaction_id"], state["document_id"]
state["confirmed_template_id"], state["confirmed_template_name"]

Writes
------
state["cross_reference_results"]  — flat list of DiscrepancyItem dicts
state["verification_report"]      — complete VerificationReport dict
state["current_step"]
Appends to state["errors"]
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_llm
from app.langgraph.schemas.coo_pacd_schemas import DiscrepancyItem, VerificationReport
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_NARRATIVE_PROMPT = """\
You are a trade compliance verification officer.

You have completed a three-section cross-reference of a Certificate of Origin (COO)
against Pre-Arrival Customs Documents (PACD) for the same transaction.

Below is the consolidated discrepancy table. Write a professional verification report:
  1. A concise 2-3 sentence SUMMARY.
  2. A detailed NARRATIVE (3-5 paragraphs) describing findings, mismatches,
     and recommended actions.

Respond in plain text with two clearly labelled sections:
SUMMARY:
<text>

NARRATIVE:
<text>
"""


def _overall_verdict(matched: int, mismatched: int, not_found: int, total: int) -> str:
    if total == 0:
        return "INCONCLUSIVE"
    if mismatched > 0:
        return "FAIL"
    if not_found == total:
        return "INCONCLUSIVE"
    return "PASS"


def _table_text(rows: list[DiscrepancyItem]) -> str:
    lines = ["Field | COO Value | PACD Value | Verdict | Source", "-" * 70]
    for row in rows:
        lines.append(
            f"{row.field_key} | {row.coo_value or '—'} | "
            f"{row.pacd_value or '—'} | {row.verdict} | {row.pacd_source_doc or '—'}"
        )
    return "\n".join(lines)


def _header_verdicts_to_discrepancy(verdicts: list[dict[str, Any]]) -> list[DiscrepancyItem]:
    items: list[DiscrepancyItem] = []
    for v in verdicts:
        try:
            items.append(DiscrepancyItem(
                field_key=v.get("field_key", "unknown"),
                coo_value=v.get("coo_value"),
                pacd_value=v.get("pacd_value"),
                verdict=v.get("verdict", "not_found_in_pacd"),  # type: ignore[arg-type]
                pacd_source_doc=v.get("pacd_source_doc") or v.get("pacd_source_filename"),
            ))
        except Exception:
            pass
    return items


def _content_verdicts_to_discrepancy(entries: list[dict[str, Any]]) -> list[DiscrepancyItem]:
    """Flatten ItemVerificationEntry list into DiscrepancyItems."""
    items: list[DiscrepancyItem] = []
    for entry in entries:
        item_num = entry.get("coo_item_number") or "?"
        source = entry.get("matched_pacd_source")
        for fv in entry.get("field_verdicts", []):
            field_name = fv.get("field_name", "field")
            try:
                items.append(DiscrepancyItem(
                    field_key=f"item_{item_num}_{field_name}",
                    coo_value=fv.get("coo_value"),
                    pacd_value=fv.get("pacd_value"),
                    verdict=fv.get("verdict", "not_found_in_pacd"),  # type: ignore[arg-type]
                    pacd_source_doc=source,
                ))
            except Exception:
                pass
    return items


def _footer_verdicts_to_discrepancy(verdicts: list[dict[str, Any]]) -> list[DiscrepancyItem]:
    items: list[DiscrepancyItem] = []
    for v in verdicts:
        try:
            items.append(DiscrepancyItem(
                field_key=v.get("field_key", "unknown"),
                coo_value=v.get("coo_value"),
                pacd_value=v.get("pacd_value"),
                verdict=v.get("verdict", "not_found_in_pacd"),  # type: ignore[arg-type]
                pacd_source_doc=v.get("pacd_source_doc"),
            ))
        except Exception:
            pass
    return items


def report_consolidation_node(state: GraphState) -> GraphState:
    """LangGraph node: merge parallel section results and generate final report.

    Runs after all three verification nodes have completed (fan-in).
    """
    logger.info("report_consolidation_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    transaction_id: str = state.get("transaction_id") or ""
    coo_doc_id: str = state.get("document_id") or ""

    header_results: list[dict] = state.get("header_verification_results") or []
    content_results: list[dict] = state.get("content_verification_results") or []
    footer_results: list[dict] = state.get("footer_verification_results") or []

    # ── Merge all sections into a flat discrepancy table ──────────────────
    all_items: list[DiscrepancyItem] = (
        _header_verdicts_to_discrepancy(header_results)
        + _content_verdicts_to_discrepancy(content_results)
        + _footer_verdicts_to_discrepancy(footer_results)
    )

    matched   = sum(1 for d in all_items if d.verdict == "match")
    mismatched = sum(1 for d in all_items if d.verdict == "mismatch")
    not_found  = sum(1 for d in all_items if d.verdict == "not_found_in_pacd")
    total      = len(all_items)
    verdict    = _overall_verdict(matched, mismatched, not_found, total)

    logger.info(
        "report_consolidation_node: total=%d matched=%d mismatched=%d not_found=%d verdict=%s",
        total, matched, mismatched, not_found, verdict,
    )

    # ── Generate narrative via Text LLM ───────────────────────────────────
    human_msg = (
        f"Transaction: {transaction_id}\n"
        f"COO Document: {coo_doc_id}\n"
        f"Confirmed Template: {state.get('confirmed_template_name') or 'Unknown'}\n"
        f"Overall Verdict: {verdict}\n"
        f"Matched: {matched}/{total}  |  Mismatched: {mismatched}  |  Not found: {not_found}\n\n"
        f"Cross-Reference Table (all sections):\n{_table_text(all_items)}\n\n"
        "Please write the verification report."
    )

    summary = ""
    narrative = ""
    try:
        llm = get_llm(temperature=0.3, max_tokens=1024)
        response = llm.invoke([
            SystemMessage(content=_NARRATIVE_PROMPT),
            HumanMessage(content=human_msg),
        ])
        raw_text = str(response.content)
        if "NARRATIVE:" in raw_text:
            parts = raw_text.split("NARRATIVE:", 1)
            narrative = parts[1].strip()
            summary = parts[0].split("SUMMARY:", 1)[-1].strip() if "SUMMARY:" in parts[0] else parts[0].strip()
        elif "SUMMARY:" in raw_text:
            summary = raw_text.split("SUMMARY:", 1)[1].strip()
        else:
            narrative = raw_text.strip()
            summary = raw_text[:300].strip()
    except Exception as exc:
        errors.append(f"report_consolidation_node: LLM narrative failed — {exc}")
        narrative = "Report narrative generation failed. Review the discrepancy table."
        summary = f"Verification {verdict}: {matched}/{total} fields matched."

    # ── Assemble VerificationReport ───────────────────────────────────────
    report = VerificationReport(
        transaction_id=transaction_id,
        coo_doc_id=coo_doc_id,
        confirmed_template_id=state.get("confirmed_template_id"),
        confirmed_template_name=state.get("confirmed_template_name"),
        discrepancy_table=all_items,
        narrative=narrative,
        summary=summary,
        total_fields=total,
        matched=matched,
        mismatched=mismatched,
        not_found=not_found,
        overall_verdict=verdict,  # type: ignore[arg-type]
    )

    return {  # type: ignore[return-value]
        **state,
        "cross_reference_results": [d.model_dump() for d in all_items],
        "verification_report": report.model_dump(),
        "errors": errors,
        "current_step": "report_consolidation_node",
    }
