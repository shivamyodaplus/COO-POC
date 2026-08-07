"""
PACDIngestionWorkflow — ingests PACD document pages into structured storage.

Graph topology
--------------
    START
      │
      ▼
  ingestion_node              validate file, assign document_id, detect MIME, load page images
      │
      ▼
  vision_extraction_node      Vision LLM: extract structured data per page
      │                       (header_fields + line_items per page)
      ▼
  pacd_layout_chunking_node   Map phase: merge pages into logical documents,
      │                       store in Postgres + index in Milvus pacd_document_chunks
      │                       (layout-aware: table chunks split by MAX_TABLE_ITEMS_PER_CHUNK)
      ▼
    END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.nodes.pacd_layout_chunking_node import pacd_layout_chunking_node
from app.langgraph.nodes.vision_extraction_node import vision_extraction_node
from app.langgraph.state import GraphState


def _abort_on_ingestion_failure(state: GraphState) -> str:
    """Short-circuit to END if ingestion failed (bad file type, no bytes)."""
    if not state.get("document_id") or state.get("mime_type") == "application/octet-stream":
        return END  # type: ignore[return-value]
    return "vision_extraction_node"


def build_pacd_ingestion_workflow() -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("ingestion_node",             ingestion_node)
    graph.add_node("vision_extraction_node",     vision_extraction_node)
    graph.add_node("pacd_layout_chunking_node",  pacd_layout_chunking_node)

    graph.add_edge(START, "ingestion_node")
    graph.add_conditional_edges(
        "ingestion_node",
        _abort_on_ingestion_failure,
        {"vision_extraction_node": "vision_extraction_node", END: END},
    )
    graph.add_edge("vision_extraction_node",    "pacd_layout_chunking_node")
    graph.add_edge("pacd_layout_chunking_node", END)

    return graph.compile()
