"""
FooterVerificationNode — verifies COO footer-level fields (totals, payment terms,
remarks, authorizations) against retrieved PACD footer section chunks.

Runs in parallel with header_verification_node and content_verification_node.

Flow
----
1. Identify footer-level fields from COO header_fields (keys that match footer
   keywords: total_, remark, terms, signature, payment, bank, etc.).
2. Read footer_queries from state["section_queries"]["footer"].
3. Search Milvus pacd_chunks (chunk_type="footer") via RRF-fused multi-query search.
4. Single LLM call → FooterVerificationResult (per-field verdicts).
5. Write results to state["footer_verification_results"].
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.structured_extraction import FooterVerificationResult
from app.langgraph.state import GraphState
from app.services import pacd_milvus_service

logger = logging.getLogger(__name__)

# Keywords that identify footer-level fields (same set as pacd_structuring_node)
_FOOTER_KEYWORDS = frozenset({
    "total_", "remark", "term", "signature", "payment", "bank",
    "seal", "stamp", "authorized", "note", "comment", "condition",
    "balance", "due", "freight", "insurance", "grand",
})


def _is_footer_field(key: str) -> bool:
    k = key.lower()
    return any(k.startswith(kw) or kw in k for kw in _FOOTER_KEYWORDS)


_SYSTEM_PROMPT = """\
You are a trade compliance expert verifying footer-level fields from a Certificate of
Origin (COO) against Pre-Arrival Customs Documents (PACD).

Footer fields include: total amounts, gross/net weights, payment terms, bank details,
remarks, authorized signatures, seal information, freight/insurance charges.

You are given:
1. COO footer fields — summary/authorization fields from the certificate.
2. Retrieved PACD footer chunks — footer passages from PACD documents for this transaction.

For EACH COO footer field, determine whether its value is supported by the PACD evidence.

Rules:
- "match": Values agree (case-insensitive, minor formatting differences ok).
- "mismatch": A related field exists in PACD but the value differs materially.
- "not_found_in_pacd": No corresponding evidence found in any PACD chunk.
- Compare semantically equivalent fields.
- Every COO footer field must appear in the output exactly once.
"""


def _format_pacd_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No PACD footer chunks retrieved."
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta_str = ""
        try:
            meta = json.loads(chunk.get("chunk_metadata", "{}"))
            if meta.get("footer_fields"):
                meta_str = "  Fields: " + ", ".join(
                    f"{k}={v}" for k, v in meta["footer_fields"].items()
                )
        except Exception:
            pass
        lines.append(
            f"[Chunk {i} | {chunk.get('doc_type', '?')} | score={chunk.get('rrf_score', 0):.4f}]\n"
            f"  {chunk.get('chunk_text', '')}{meta_str}"
        )
    return "\n\n".join(lines)


def footer_verification_node(state: GraphState) -> GraphState:
    """LangGraph node: verify COO footer fields against retrieved PACD footer chunks.

    Reads
    -----
    state["section_queries"]["footer"]
    state["coo_extracted_data"]["header_fields"]  (footer-keyed subset)
    state["transaction_id"]

    Writes
    ------
    state["footer_verification_results"]  — list of FooterFieldVerdict dicts
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("footer_verification_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    transaction_id: str = state.get("transaction_id") or ""
    coo_data: dict[str, Any] = state.get("coo_extracted_data") or {}
    all_coo_headers: dict[str, str] = coo_data.get("header_fields", {})
    section_queries: dict[str, Any] = state.get("section_queries") or {}
    footer_queries: list[str] = section_queries.get("footer", [])

    # Extract footer-keyed fields from COO headers
    coo_footer_fields = {k: v for k, v in all_coo_headers.items() if _is_footer_field(k) and v}

    if not coo_footer_fields:
        logger.info("footer_verification_node: no footer fields found in COO — skipping")
        return {"footer_verification_results": []}  # type: ignore[return-value]

    # ── Retrieve PACD footer chunks ───────────────────────────────────────
    chunks: list[dict[str, Any]] = []
    if footer_queries and transaction_id:
        try:
            chunks = pacd_milvus_service.search_section_chunks(
                section="footer",
                queries=footer_queries,
                transaction_id=transaction_id,
                top_k=10,
                top_t=6,
            )
            logger.info("footer_verification_node: retrieved %d footer chunks", len(chunks))
        except Exception as exc:
            errors.append(f"footer_verification_node: Milvus search failed — {exc}")
            logger.warning("footer_verification_node: Milvus error: %s", exc)

    # ── Build LLM context ─────────────────────────────────────────────────
    coo_block = "\n".join(f"  {k}: {v}" for k, v in coo_footer_fields.items())
    pacd_block = _format_pacd_chunks(chunks)

    user_msg = (
        f"COO Footer Fields:\n{coo_block}\n\n"
        f"Retrieved PACD Footer Chunks:\n{pacd_block}"
    )

    # ── LLM verification call ─────────────────────────────────────────────
    llm = get_structured_llm(FooterVerificationResult, temperature=0.0)
    try:
        result: FooterVerificationResult = llm.invoke([  # type: ignore[assignment]
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        verdicts = [v.model_dump() for v in result.verdicts]
        logger.info(
            "footer_verification_node: %d verdicts produced",
            len(verdicts),
        )
    except Exception as exc:
        logger.warning("footer_verification_node: LLM error: %s", exc)
        verdicts = [
            {
                "field_key": k, "coo_value": v, "pacd_value": None,
                "verdict": "not_found_in_pacd", "pacd_source_doc": None,
            }
            for k, v in coo_footer_fields.items()
        ]

    return {"footer_verification_results": verdicts}  # type: ignore[return-value]
