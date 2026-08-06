"""
PACDIngestionWorkflow — ingests PACD document pages into structured storage.

Graph topology
--------------
    START
      │
      ▼
  ingestion_node           validate file, assign document_id, detect MIME, load page images
      │
      ▼
  vision_extraction_node   Vision LLM: extract structured data per page
      │                    (uses with_structured_output for PACD → header_fields + line_items)
      ▼
  pacd_structuring_node    merge pages into logical documents, store in Postgres + Milvus
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
        "structured_pages": [],
        "page_images": [],
        "extracted_fields": {},
        "retrieved_templates": [],
        "template_attributes": {},
        "cross_reference_results": [],
        "cross_reference_details": {},
        "verification_report": {},
        "coo_extracted_fields": {},
        "coo_extracted_data": {},
    })
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.nodes.pacd_structuring_node import pacd_structuring_node
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
    graph.add_node("pacd_structuring_node", pacd_structuring_node)

    graph.add_edge(START, "ingestion_node")
    graph.add_conditional_edges(
        "ingestion_node",
        _abort_on_ingestion_failure,
        {"vision_extraction_node": "vision_extraction_node", END: END},
    )
    graph.add_edge("vision_extraction_node", "pacd_structuring_node")
    graph.add_edge("pacd_structuring_node", END)

    return graph.compile()
