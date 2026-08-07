"""
COOVerificationWorkflow — Map-Reduce RAG pipeline for COO cross-reference.

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
  coo_extraction_node         Vision LLM: extract COO header_fields + line_items
      │
      ▼
  multi_query_expansion_node  Text LLM: expand COO entities into 4 query categories
      │                       (item / header / footer / hs queries)
      ▼
  rag_retrieval_node          Hybrid BGE-M3 search on pacd_document_chunks
      │                       + table isolation (fetch all sibling table chunks)
      ▼
  context_assembly_node       Group and format retrieved chunks into structured XML
      │
      ▼
  llm_critic_node             Single Text LLM call: verify COO vs condensed context
      │                       → VerificationReport (same shape as before)
      ▼
    END

This replaces the old 3-way parallel verification (section_query_node +
header/content/footer_verification_node + report_consolidation_node) with
a Map-Reduce RAG approach that handles large PACD documents (50+ pages)
without context window exhaustion.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.coo_extraction_node import coo_extraction_node
from app.langgraph.nodes.context_assembly_node import context_assembly_node
from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.nodes.llm_critic_node import llm_critic_node
from app.langgraph.nodes.multi_query_expansion_node import multi_query_expansion_node
from app.langgraph.nodes.rag_retrieval_node import rag_retrieval_node
from app.langgraph.nodes.template_confirmation_node import template_confirmation_node
from app.langgraph.nodes.template_retrieval_node import template_retrieval_node
from app.langgraph.state import GraphState


def _after_ingestion(state: GraphState) -> str:
    if not state.get("document_id") or state.get("mime_type") == "application/octet-stream":
        return END  # type: ignore[return-value]
    return "template_retrieval_node"


def _after_retrieval(state: GraphState) -> str:
    if not state.get("retrieved_templates"):
        return "coo_extraction_node"
    return "template_confirmation_node"


def build_coo_verification_workflow() -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("ingestion_node",              ingestion_node)
    graph.add_node("template_retrieval_node",      template_retrieval_node)
    graph.add_node("template_confirmation_node",   template_confirmation_node)
    graph.add_node("coo_extraction_node",          coo_extraction_node)
    graph.add_node("multi_query_expansion_node",   multi_query_expansion_node)
    graph.add_node("rag_retrieval_node",           rag_retrieval_node)
    graph.add_node("context_assembly_node",        context_assembly_node)
    graph.add_node("llm_critic_node",              llm_critic_node)

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
            "coo_extraction_node":        "coo_extraction_node",
        },
    )
    graph.add_edge("template_confirmation_node",  "coo_extraction_node")
    graph.add_edge("coo_extraction_node",         "multi_query_expansion_node")
    graph.add_edge("multi_query_expansion_node",  "rag_retrieval_node")
    graph.add_edge("rag_retrieval_node",          "context_assembly_node")
    graph.add_edge("context_assembly_node",       "llm_critic_node")
    graph.add_edge("llm_critic_node",             END)

    return graph.compile()


