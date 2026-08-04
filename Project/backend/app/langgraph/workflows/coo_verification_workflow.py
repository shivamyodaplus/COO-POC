"""
COOVerificationWorkflow — full end-to-end COO cross-reference pipeline.

Graph topology
--------------
    START
      │
      ▼
  ingestion_node              validate file, assign document_id, load page images
      │ (abort on bad file)
      ▼
  template_retrieval_node     top-3 visual template candidates from Milvus
      │ (abort if no candidates)
      ▼
  template_confirmation_node  Vision LLM: confirm which template the COO used
      │ (abort if no match)
      ▼
  coo_extraction_node         Vision LLM: extract COO fields using template schema
      │
      ▼
  cross_reference_node        compare COO fields against PACD knowledge base
      │
      ▼
  report_generation_node      Text LLM: build discrepancy table + narrative
      │
      ▼
    END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.coo_extraction_node import coo_extraction_node
from app.langgraph.nodes.cross_reference_node import cross_reference_node
from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.nodes.report_generation_node import report_generation_node
from app.langgraph.nodes.template_confirmation_node import template_confirmation_node
from app.langgraph.nodes.template_retrieval_node import template_retrieval_node
from app.langgraph.state import GraphState


def _after_ingestion(state: GraphState) -> str:
    if not state.get("document_id") or state.get("mime_type") == "application/octet-stream":
        return END  # type: ignore[return-value]
    return "template_retrieval_node"


def _after_retrieval(state: GraphState) -> str:
    if not state.get("retrieved_templates"):
        return "report_generation_node"  # will produce INCONCLUSIVE with no template
    return "template_confirmation_node"


def _after_confirmation(state: GraphState) -> str:
    # Even if no template was confirmed, we still attempt extraction (general mode)
    return "coo_extraction_node"


def build_coo_verification_workflow() -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("ingestion_node",              ingestion_node)
    graph.add_node("template_retrieval_node",      template_retrieval_node)
    graph.add_node("template_confirmation_node",   template_confirmation_node)
    graph.add_node("coo_extraction_node",          coo_extraction_node)
    graph.add_node("cross_reference_node",         cross_reference_node)
    graph.add_node("report_generation_node",       report_generation_node)

    graph.add_edge(START, "ingestion_node")
    graph.add_conditional_edges(
        "ingestion_node",
        _after_ingestion,
        {"template_retrieval_node": "template_retrieval_node", END: END},
    )
    graph.add_conditional_edges(
        "template_retrieval_node",
        _after_retrieval,
        {
            "template_confirmation_node": "template_confirmation_node",
            "report_generation_node": "report_generation_node",
        },
    )
    graph.add_conditional_edges(
        "template_confirmation_node",
        _after_confirmation,
        {"coo_extraction_node": "coo_extraction_node"},
    )
    graph.add_edge("coo_extraction_node",  "cross_reference_node")
    graph.add_edge("cross_reference_node", "report_generation_node")
    graph.add_edge("report_generation_node", END)

    return graph.compile()
