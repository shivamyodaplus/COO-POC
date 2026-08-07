"""
LLMCriticNode — Reduce phase, step 4 (final) of the Map-Reduce RAG pipeline.

Single LLM call that verifies the COO against the condensed PACD reference
context assembled by ``context_assembly_node``.

By the time this node runs, the 250+ pages of raw PACD documents have been
reduced to a hyper-focused context of 5-18 chunks covering only the sections
that are directly relevant to the COO's entities.  The LLM is therefore never
asked to search a massive context window — it only validates what is in front
of it.

Outputs the same ``VerificationReport`` shape as the old
``report_consolidation_node``, ensuring the frontend remains unchanged.

Prompt structure
----------------
    <validation_task> ... </validation_task>

    <target_coo>
      [COO header + line items as markdown]
    </target_coo>

    <filtered_reference_context>
      <doc id="commercial_invoice" relevant_pages="[4,5,6]">
        ...
      </doc>
    </filtered_reference_context>

    <verification_instruction>
      [detailed field-by-field comparison instructions]
    </verification_instruction>

The LLM is asked to produce a ``CriticOutput`` structured object with:
  - ``discrepancy_table``: one ``DiscrepancyRow`` per verified field
  - ``summary``, ``narrative``: text sections
  - ``overall_verdict``: "PASS" | "FAIL" | "INCONCLUSIVE"
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.coo_pacd_schemas import DiscrepancyItem, VerificationReport
from app.langgraph.schemas.structured_extraction import CriticOutput, DiscrepancyRow
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior trade compliance verification officer specialising in
Certificate of Origin (COO) cross-reference checks.

Your task: given the COO document and filtered reference sections from the
corresponding Pre-Arrival Customs Documents (PACD), verify whether the COO
fields match the PACD data.

VERIFICATION RULES
------------------
1. Compare EVERY COO header field (exporter, consignee, invoice number, date,
   ports, country of origin, etc.) against the PACD reference context.
2. For EVERY COO line item, compare: hs_code, description, quantity, unit,
   value, weight, origin_country.
3. HS code matching rule: "7318.15" matches "7318.15.00" (prefix match is OK).
4. Quantity/weight matching: treat "500 PCS" and "500 pieces" as equivalent;
   numeric values must match within 1% tolerance.
5. If a field is present in the PACD but with a minor spelling difference that
   clearly refers to the same entity, verdict = "match".
6. If a field value clearly differs, verdict = "mismatch".
7. If the PACD context contains no evidence for a COO field, verdict =
   "not_found_in_pacd".
8. Note the PACD document type (e.g. "commercial_invoice") in pacd_source_doc.

OVERALL VERDICT RULES
---------------------
- "PASS": all verified fields are "match" (may have some "not_found_in_pacd").
- "FAIL": at least one field is "mismatch".
- "INCONCLUSIVE": no PACD context was retrieved, or all fields are
  "not_found_in_pacd".

OUTPUT
------
Return a structured CriticOutput with:
- discrepancy_table: one row per field verified (field_key in format like
  "exporter_name", "item_1_hs_code", "item_2_description", etc.)
- summary: 2-3 sentence executive summary
- narrative: 3-5 paragraph detailed narrative with evidence citations
- overall_verdict: "PASS", "FAIL", or "INCONCLUSIVE"
"""

_VERIFICATION_INSTRUCTION = """\
<verification_instruction>
Verify the Target COO against the Filtered Reference Context above.

1. For each COO header field, find the matching value in the PACD docs and
   record the verdict (match / mismatch / not_found_in_pacd) with evidence.

2. For each COO line item:
   - Match by HS code first (prefix match allowed), then by description.
   - Verify: hs_code, description, quantity, unit, value, weight, origin_country.
   - Use field_key format: "item_{N}_{field}" e.g. "item_1_hs_code".

3. If any PACD sections are missing or truncated, note this in the narrative
   and set affected fields to "not_found_in_pacd".

4. Determine overall_verdict per the rules above.
</verification_instruction>
"""


def _verdict_to_literal(v: str) -> Literal["match", "mismatch", "not_found_in_pacd", "not_in_coo"]:
    v = v.lower().strip()
    if v == "match":
        return "match"
    if v == "mismatch":
        return "mismatch"
    if v in {"not_in_coo"}:
        return "not_in_coo"
    return "not_found_in_pacd"


def _overall_verdict_literal(v: str) -> Literal["PASS", "FAIL", "INCONCLUSIVE"]:
    v = v.upper().strip()
    if v == "PASS":
        return "PASS"
    if v == "FAIL":
        return "FAIL"
    return "INCONCLUSIVE"


def llm_critic_node(state: GraphState) -> GraphState:
    """LangGraph node: single LLM call to verify COO against condensed context.

    Reads
    -----
    state["assembled_context"]       — XML string from context_assembly_node
    state["coo_extracted_data"]      — {"header_fields": {}, "line_items": [...]}
    state["transaction_id"]
    state["document_id"]
    state["confirmed_template_id"]
    state["confirmed_template_name"]

    Writes
    ------
    state["verification_report"]     — VerificationReport dict (same shape as
                                       old report_consolidation_node output)
    state["cross_reference_results"] — flat list of DiscrepancyItem dicts
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("llm_critic_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    assembled_context = state.get("assembled_context") or ""
    coo_data: dict[str, Any] = state.get("coo_extracted_data") or {}
    transaction_id = state.get("transaction_id") or ""
    document_id = state.get("document_id") or ""
    template_id = state.get("confirmed_template_id")
    template_name = state.get("confirmed_template_name")

    # Build the human message combining context + instructions
    human_content = (
        f"{assembled_context}\n\n"
        f"{_VERIFICATION_INSTRUCTION}"
    )

    critic_result: CriticOutput | None = None
    try:
        llm = get_structured_llm(CriticOutput)
        critic_result = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])
        logger.info(
            "llm_critic_node: verdict=%s fields=%d",
            critic_result.overall_verdict,
            len(critic_result.discrepancy_table),
        )
    except Exception as exc:
        errors.append(f"llm_critic_node: LLM call failed — {exc}")
        logger.error("llm_critic_node: LLM error: %s", exc)

    # ── Build discrepancy table ────────────────────────────────────────────
    discrepancy_items: list[DiscrepancyItem] = []
    if critic_result and critic_result.discrepancy_table:
        for row in critic_result.discrepancy_table:
            if isinstance(row, dict):
                row = DiscrepancyRow(**row)
            try:
                discrepancy_items.append(DiscrepancyItem(
                    field_key=row.field_key,
                    coo_value=row.coo_value,
                    pacd_value=row.pacd_value,
                    verdict=_verdict_to_literal(row.verdict),
                    pacd_source_doc=row.pacd_source_doc,
                ))
            except Exception as e:
                logger.warning("llm_critic_node: skipping malformed row %r — %s", row, e)

    # ── Compute aggregate counts ──────────────────────────────────────────
    total = len(discrepancy_items)
    matched   = sum(1 for d in discrepancy_items if d.verdict == "match")
    mismatched = sum(1 for d in discrepancy_items if d.verdict == "mismatch")
    not_found  = sum(1 for d in discrepancy_items if d.verdict == "not_found_in_pacd")

    # Derive overall verdict if LLM didn't produce one
    if critic_result:
        overall = _overall_verdict_literal(critic_result.overall_verdict)
    else:
        overall = "INCONCLUSIVE"
        if mismatched > 0:
            overall = "FAIL"
        elif total > 0 and matched > 0:
            overall = "PASS"

    summary   = (critic_result.summary   if critic_result else "Verification could not be completed — LLM error.")
    narrative = (critic_result.narrative if critic_result else "The LLM critic node encountered an error. Please review the errors list.")

    # ── Assemble VerificationReport ───────────────────────────────────────
    report = VerificationReport(
        transaction_id=transaction_id,
        coo_doc_id=document_id,
        confirmed_template_id=template_id,
        confirmed_template_name=template_name,
        discrepancy_table=discrepancy_items,
        summary=summary,
        narrative=narrative,
        total_fields=total,
        matched=matched,
        mismatched=mismatched,
        not_found=not_found,
        overall_verdict=overall,
    )
    report_dict = report.model_dump()

    return {
        **state,
        "verification_report":     report_dict,
        "cross_reference_results": [d.model_dump() for d in discrepancy_items],
        "errors":                  errors,
        "current_step":            "llm_critic_node",
    }  # type: ignore[return-value]
