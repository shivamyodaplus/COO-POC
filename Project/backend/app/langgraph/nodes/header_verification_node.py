"""
HeaderVerificationNode — verifies COO header fields against retrieved PACD
header section chunks.

Runs in parallel with content_verification_node and footer_verification_node.

Flow
----
1. Read header_queries from state["section_queries"]["header"].
2. Search Milvus pacd_chunks (chunk_type="header") using RRF-fused multi-query search.
3. Build context: COO header fields + retrieved PACD header chunk texts.
4. Single LLM call → HeaderVerificationResult (per-field verdicts).
5. Write results to state["header_verification_results"].
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.structured_extraction import HeaderVerificationResult
from app.langgraph.state import GraphState
from app.services import pacd_milvus_service

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a trade compliance expert verifying a Certificate of Origin (COO) against
Pre-Arrival Customs Documents (PACD).

You are given:
1. COO header fields — extracted from the certificate being verified.
2. Retrieved PACD header chunks — text passages from PACD documents for this transaction.

For EACH COO header field, determine whether its value is supported by the PACD evidence.

Rules:
- "match": The COO value matches a corresponding PACD value
  (case-insensitive, ignore minor whitespace / formatting differences,
   abbreviations like "Co." = "Company" are acceptable).
- "mismatch": A related field exists in PACD but the value differs materially.
- "not_found_in_pacd": No corresponding evidence found in any PACD chunk.
- Compare semantically equivalent fields (e.g. "shipper" = "exporter", "buyer" = "consignee").
- Every COO header field must appear in the output exactly once.
- Use the chunk_metadata JSON to find precise values when the chunk_text is ambiguous.
"""


def _format_pacd_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No PACD header chunks retrieved."
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta_str = ""
        try:
            meta = json.loads(chunk.get("chunk_metadata", "{}"))
            if meta.get("header_fields"):
                meta_str = "  Fields: " + ", ".join(
                    f"{k}={v}" for k, v in meta["header_fields"].items()
                )
        except Exception:
            pass
        lines.append(
            f"[Chunk {i} | {chunk.get('doc_type', '?')} | score={chunk.get('rrf_score', 0):.4f}]\n"
            f"  {chunk.get('chunk_text', '')}{meta_str}"
        )
    return "\n\n".join(lines)


def header_verification_node(state: GraphState) -> GraphState:
    """LangGraph node: verify COO header fields against retrieved PACD header chunks.

    Reads
    -----
    state["section_queries"]["header"]
    state["coo_extracted_data"]["header_fields"]
    state["transaction_id"]

    Writes
    ------
    state["header_verification_results"]  — list of HeaderFieldVerdict dicts
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("header_verification_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    transaction_id: str = state.get("transaction_id") or ""
    coo_data: dict[str, Any] = state.get("coo_extracted_data") or {}
    coo_headers: dict[str, str] = coo_data.get("header_fields", {})
    section_queries: dict[str, Any] = state.get("section_queries") or {}
    header_queries: list[str] = section_queries.get("header", [])

    if not coo_headers:
        logger.warning("header_verification_node: no COO header fields to verify")
        return {"header_verification_results": []}  # type: ignore[return-value]

    # ── Retrieve PACD header chunks ───────────────────────────────────────
    chunks: list[dict[str, Any]] = []
    if header_queries and transaction_id:
        try:
            chunks = pacd_milvus_service.search_section_chunks(
                section="header",
                queries=header_queries,
                transaction_id=transaction_id,
                top_k=10,
                top_t=8,
            )
            logger.info("header_verification_node: retrieved %d header chunks", len(chunks))
        except Exception as exc:
            errors.append(f"header_verification_node: Milvus search failed — {exc}")
            logger.warning("header_verification_node: Milvus error: %s", exc)

    # ── Build LLM context ─────────────────────────────────────────────────
    coo_block = "\n".join(f"  {k}: {v}" for k, v in coo_headers.items())
    pacd_block = _format_pacd_chunks(chunks)

    user_msg = (
        f"COO Header Fields:\n{coo_block}\n\n"
        f"Retrieved PACD Header Chunks:\n{pacd_block}"
    )

    # ── LLM verification call ─────────────────────────────────────────────
    llm = get_structured_llm(HeaderVerificationResult, temperature=0.0)
    try:
        result: HeaderVerificationResult = llm.invoke([  # type: ignore[assignment]
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        verdicts = [v.model_dump() for v in result.verdicts]
        logger.info(
            "header_verification_node: %d verdicts produced",
            len(verdicts),
        )
    except Exception as exc:
        errors.append(f"header_verification_node: LLM failed — {exc}")
        logger.warning("header_verification_node: LLM error: %s", exc)
        # Fallback: mark all fields as not_found
        verdicts = [
            {
                "field_key": k, "coo_value": v, "pacd_value": None,
                "verdict": "not_found_in_pacd", "pacd_source_doc": None,
                "pacd_source_filename": None,
            }
            for k, v in coo_headers.items()
        ]

    # Parallel nodes MUST return only their own field — no **state spread.
    # Spreading **state from concurrent branches causes LangGraph InvalidUpdateError
    # because all three branches write to the same shared keys simultaneously.
    return {"header_verification_results": verdicts}  # type: ignore[return-value]
