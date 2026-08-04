"""
PACDIngestionWorkflow — ingests PACD document pages into the knowledge base.

Graph topology
--------------
    START
      │
      ▼
  ingestion_node         validate file, assign document_id, detect MIME, load page images
      │
      ▼
  vision_extraction_node  Vision LLM: extract key-value pairs from every page
      │
      ▼
  pacd_indexing_node      store embeddings + KV data in Milvus + Postgres
      │
      ▼
    END

Usage
-----
    from app.langgraph.workflows.pacd_ingestion_workflow import build_pacd_ingestion_workflow
    from app.langgraph.adapters.langgraph_adapter import LangGraphAdapter

    graph   = build_pacd_ingestion_workflow()
    adapter = LangGraphAdapter(graph, "pacd_ingestion_workflow")

    result = await adapter.invoke({
        "raw_bytes": file_bytes,
        "filename": "pacd_doc.pdf",
        "user_id": "user123",
        "transaction_id": "tx-uuid",
        "doc_category": "pacd",
        "messages": [],
        "errors": [],
        "extracted_kv_pairs": [],
        "page_images": [],
        "extracted_fields": {},
        "retrieved_templates": [],
        "template_attributes": {},
        "cross_reference_results": [],
        "verification_report": {},
        "coo_extracted_fields": {},
    })
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.nodes.pacd_indexing_node import pacd_indexing_node
from app.langgraph.nodes.vision_extraction_node import vision_extraction_node
from app.langgraph.state import GraphState


def _abort_on_ingestion_failure(state: GraphState) -> str:
    """Short-circuit to END if ingestion failed (bad file type, no bytes)."""
    if not state.get("document_id") or state.get("mime_type") == "application/octet-stream":
        return END  # type: ignore[return-value]
    return "vision_extraction_node"


def build_pacd_ingestion_workflow() -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("ingestion_node", ingestion_node)
    graph.add_node("vision_extraction_node", vision_extraction_node)
    graph.add_node("pacd_indexing_node", pacd_indexing_node)

    graph.add_edge(START, "ingestion_node")
    graph.add_conditional_edges(
        "ingestion_node",
        _abort_on_ingestion_failure,
        {"vision_extraction_node": "vision_extraction_node", END: END},
    )
    graph.add_edge("vision_extraction_node", "pacd_indexing_node")
    graph.add_edge("pacd_indexing_node", END)

    return graph.compile()
