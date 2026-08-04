"""
DocumentWorkflow — orchestrates ingestion → classification → extraction.

Graph topology
--------------
    START
      │
      ▼
  ingestion_node          validate file, assign document_id, detect MIME
      │
      ▼
  classification_node     LLM: identify country + doc_type
      │
      ▼
  extraction_node         LLM: extract structured fields
      │
      ▼
    END

Conditional edges
-----------------
After ``ingestion_node``: if errors are present the graph can route to END
early (``_should_continue`` guard).  This prevents the LLM nodes from being
called with invalid data.

Usage
-----
    from app.langgraph.workflows.document_workflow import build_document_workflow
    from app.langgraph.adapters.langgraph_adapter import LangGraphAdapter

    graph   = build_document_workflow()           # CompiledStateGraph
    adapter = LangGraphAdapter(graph, "document_workflow")

    result = await adapter.invoke({
        "raw_bytes": file_bytes,
        "filename": "passport_scan.pdf",
        "messages": [],
        "extracted_fields": {},
        "retrieved_templates": [],
        "errors": [],
    })
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.classification_node import classification_node
from app.langgraph.nodes.extraction_node import extraction_node
from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.state import GraphState


# --------------------------------------------------------------------------- #
# Conditional routing helpers                                                  #
# --------------------------------------------------------------------------- #

def _should_continue_after_ingestion(state: GraphState) -> str:
    """Route to END if ingestion encountered fatal errors, else classify."""
    errors = state.get("errors") or []
    mime = state.get("mime_type") or ""
    if not state.get("document_id") or mime == "application/octet-stream":
        return END  # type: ignore[return-value]
    return "classification_node"


# --------------------------------------------------------------------------- #
# Graph builder                                                                #
# --------------------------------------------------------------------------- #

def build_document_workflow() -> CompiledStateGraph:
    """Build and compile the document processing workflow.

    Returns a ``CompiledStateGraph`` ready to be wrapped in a
    ``LangGraphAdapter`` or invoked directly for testing.
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("ingestion_node", ingestion_node)
    graph.add_node("classification_node", classification_node)
    graph.add_node("extraction_node", extraction_node)

    # Entry point
    graph.add_edge(START, "ingestion_node")

    # Conditional edge: route to END on ingestion failure, else classify
    graph.add_conditional_edges(
        "ingestion_node",
        _should_continue_after_ingestion,
        {
            "classification_node": "classification_node",
            END: END,
        },
    )

    # Linear edges for the happy path
    graph.add_edge("classification_node", "extraction_node")
    graph.add_edge("extraction_node", END)

    return graph.compile()
