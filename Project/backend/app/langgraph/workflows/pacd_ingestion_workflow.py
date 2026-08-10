"""
PACDIngestionWorkflow — ingests PACD document pages into Milvus via VLM chunking.

Graph topology (2-Tier GraphRAG)
--------------------------------
    START
      │
      ▼
  ingestion_node        validate file, assign document_id, detect MIME, load page images
      │
      ▼
  vlm_chunking_node     Vision LLM: extract logical chunks (line items + header fields),
      │                 embed with BGE-M3, insert into coo_reference_chunks collection
      ▼
    END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.nodes.vlm_chunking_node import vlm_chunking_node
from app.langgraph.state import GraphState


def _abort_on_ingestion_failure(state: GraphState) -> str:
    """Short-circuit to END if ingestion failed (bad file type, no bytes)."""
    if not state.get("document_id") or state.get("mime_type") == "application/octet-stream":
        return END  # type: ignore[return-value]
    return "vlm_chunking_node"


def build_pacd_ingestion_workflow() -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("ingestion_node",    ingestion_node)
    graph.add_node("vlm_chunking_node", vlm_chunking_node)

    graph.add_edge(START, "ingestion_node")
    graph.add_conditional_edges(
        "ingestion_node",
        _abort_on_ingestion_failure,
        {"vlm_chunking_node": "vlm_chunking_node", END: END},
    )
    graph.add_edge("vlm_chunking_node", END)

    return graph.compile()

