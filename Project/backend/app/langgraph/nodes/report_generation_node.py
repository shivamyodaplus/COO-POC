"""
ReportGenerationNode — final node in the COO verification workflow.

Takes the cross-reference results and asks the Text LLM to produce:
  1. A structured discrepancy table (already built by CrossReferenceNode).
  2. A plain-language narrative summary describing the overall verdict
     and highlighting any mismatches.
  3. An ``overall_verdict``: PASS (no mismatches), FAIL (≥1 mismatch), or
     INCONCLUSIVE (no PACD data found for comparison).
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_llm
from app.langgraph.schemas.coo_pacd_schemas import (
    DiscrepancyItem,
    ReportGenerationInput,
    ReportGenerationOutput,
    VerificationReport,
)
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a trade compliance verification officer.
You have cross-referenced a Certificate of Origin (COO) document against PACD
(Pre-Arrival Customs Declaration) documents for the same transaction.

Below is the cross-reference result table. Write a professional verification report:
  1. A concise 2-3 sentence SUMMARY paragraph.
  2. A detailed NARRATIVE (3-5 paragraphs) describing findings, any mismatches,
     and what action the user should take.

Respond in plain text with two clearly labelled sections:
SUMMARY:
<summary text>

NARRATIVE:
<detailed narrative text>
"""


def _overall_verdict(matched: int, mismatched: int, not_found: int, total: int) -> str:
    if total == 0:
        return "INCONCLUSIVE"
    if mismatched > 0:
        return "FAIL"
    if not_found == total:
        return "INCONCLUSIVE"
    return "PASS"


def _build_table_text(discrepancy_table: list[dict]) -> str:
    lines = ["Field | COO Value | PACD Value | Verdict"]
    lines.append("-" * 60)
    for row in discrepancy_table:
        lines.append(
            f"{row.get('field_key')} | {row.get('coo_value')} | "
            f"{row.get('pacd_value', 'N/A')} | {row.get('verdict')}"
        )
    return "\n".join(lines)


def report_generation_node(state: GraphState) -> GraphState:
    """LangGraph node: generate the final verification report via Text LLM.

    Reads
    -----
    state["cross_reference_results"], state["transaction_id"],
    state["document_id"], state["confirmed_template_id"],
    state["confirmed_template_name"]

    Writes
    ------
    state["verification_report"]  — complete VerificationReport dict
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("report_generation_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    raw_results: list[dict] = state.get("cross_reference_results") or []
    transaction_id = state.get("transaction_id") or ""
    coo_doc_id = state.get("document_id") or ""

    # Re-hydrate DiscrepancyItem objects for counting
    discrepancy_table: list[DiscrepancyItem] = []
    for row in raw_results:
        try:
            discrepancy_table.append(DiscrepancyItem(**row))
        except Exception:
            pass

    matched = sum(1 for d in discrepancy_table if d.verdict == "match")
    mismatched = sum(1 for d in discrepancy_table if d.verdict == "mismatch")
    not_found = sum(1 for d in discrepancy_table if d.verdict == "not_found_in_pacd")
    total = len(discrepancy_table)
    verdict = _overall_verdict(matched, mismatched, not_found, total)

    try:
        node_input = ReportGenerationInput(
            transaction_id=transaction_id,
            coo_doc_id=coo_doc_id,
            confirmed_template_id=state.get("confirmed_template_id"),
            confirmed_template_name=state.get("confirmed_template_name"),
            discrepancy_table=discrepancy_table,
            total_fields=total,
            matched=matched,
            mismatched=mismatched,
            not_found=not_found,
        )
    except Exception as exc:
        errors.append(f"report_generation_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "report_generation_node"}  # type: ignore[return-value]

    # --- LLM call for narrative -----------------------------------------------
    table_text = _build_table_text(raw_results)
    human_msg = (
        f"Transaction: {transaction_id}\n"
        f"COO Document: {coo_doc_id}\n"
        f"Confirmed Template: {node_input.confirmed_template_name or 'Unknown'}\n"
        f"Overall verdict: {verdict}\n"
        f"Matched: {matched}/{total}  |  Mismatched: {mismatched}  |  Not found in PACD: {not_found}\n\n"
        f"Cross-Reference Table:\n{table_text}\n\n"
        "Please write the verification report."
    )

    summary = ""
    narrative = ""

    try:
        llm = get_llm(temperature=0.3, max_tokens=1024)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ])
        raw_text = str(response.content)

        # Parse SUMMARY / NARRATIVE sections
        if "NARRATIVE:" in raw_text:
            parts = raw_text.split("NARRATIVE:", 1)
            narrative = parts[1].strip()
            if "SUMMARY:" in parts[0]:
                summary = parts[0].split("SUMMARY:", 1)[1].strip()
            else:
                summary = parts[0].strip()
        elif "SUMMARY:" in raw_text:
            summary = raw_text.split("SUMMARY:", 1)[1].strip()
        else:
            narrative = raw_text.strip()
            summary = raw_text[:300].strip()

    except Exception as exc:
        errors.append(f"report_generation_node: LLM call failed — {exc}")
        narrative = "Report narrative generation failed. Please review the discrepancy table manually."
        summary = f"Verification {verdict}: {matched}/{total} fields matched."

    report = VerificationReport(
        transaction_id=transaction_id,
        coo_doc_id=coo_doc_id,
        confirmed_template_id=node_input.confirmed_template_id,
        confirmed_template_name=node_input.confirmed_template_name,
        discrepancy_table=discrepancy_table,
        narrative=narrative,
        summary=summary,
        total_fields=total,
        matched=matched,
        mismatched=mismatched,
        not_found=not_found,
        overall_verdict=verdict,  # type: ignore[arg-type]
    )

    output = ReportGenerationOutput(report=report, errors=errors)
    logger.info("report_generation_node: verdict=%s", verdict)

    return {  # type: ignore[return-value]
        **state,
        "verification_report": output.report.model_dump(),
        "errors": output.errors,
        "current_step": "report_generation_node",
    }
