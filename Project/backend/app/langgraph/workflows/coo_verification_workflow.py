"""
COOVerificationWorkflow — 2-Tier GraphRAG pipeline for COO cross-reference.

Graph topology
--------------
    START
      │
      ▼
  ingestion_node              validate file, assign document_id, load page images
      │ (abort on bad file)
      ▼
  template_retrieval_node     visual template candidates from Milvus (Qwen3-VL)
      │
      ▼
  template_confirmation_node  Vision LLM: confirm template match (skipped if no candidates)
      │
      ▼
  coo_transcription_node      Vision LLM: transcribe COO text + generate validation queries
      │
      ▼
  parallel_search_node        BGE-M3 embed queries → parallel hybrid search on
      │                       coo_reference_chunks (filter: transaction_id)
      ▼
  validation_node             Text LLM: per-attribute verification (PASS/FAIL/UNVERIFIABLE)
      │                       → ValidationReport
      ▼
    END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.coo_transcription_node import coo_transcription_node
from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.nodes.parallel_search_node import parallel_search_node
from app.langgraph.nodes.template_confirmation_node import template_confirmation_node
from app.langgraph.nodes.template_retrieval_node import template_retrieval_node
from app.langgraph.nodes.validation_node import validation_node
from app.langgraph.state import GraphState


def _after_ingestion(state: GraphState) -> str:
    if not state.get("document_id") or state.get("mime_type") == "application/octet-stream":
        return END  # type: ignore[return-value]
    return "template_retrieval_node"


def _after_retrieval(state: GraphState) -> str:
    if not state.get("retrieved_templates"):
        return "coo_transcription_node"
    return "template_confirmation_node"


def build_coo_verification_workflow() -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("ingestion_node",              ingestion_node)
    graph.add_node("template_retrieval_node",      template_retrieval_node)
    graph.add_node("template_confirmation_node",   template_confirmation_node)
    graph.add_node("coo_transcription_node",       coo_transcription_node)
    graph.add_node("parallel_search_node",         parallel_search_node)
    graph.add_node("validation_node",              validation_node)

    # ── Edges ─────────────────────────────────────────────────────────────
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
            "coo_transcription_node":     "coo_transcription_node",
        },
    )
    graph.add_edge("template_confirmation_node",  "coo_transcription_node")
    graph.add_edge("coo_transcription_node",      "parallel_search_node")
    graph.add_edge("parallel_search_node",        "validation_node")
    graph.add_edge("validation_node",             END)

    return graph.compile()
