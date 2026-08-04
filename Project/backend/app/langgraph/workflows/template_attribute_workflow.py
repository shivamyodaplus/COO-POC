"""
TemplateAttributeWorkflow — extracts the field schema from a template image.

Runs once when a template is uploaded.  The output (template_attributes dict)
is persisted back to the ``visual_templates.extracted_attributes`` column by
the calling service so future COO extractions can use it as a field schema.

Graph topology
--------------
    START → ingestion_node → template_attribute_node → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph.nodes.ingestion_node import ingestion_node
from app.langgraph.nodes.template_attribute_node import template_attribute_node
from app.langgraph.state import GraphState


def build_template_attribute_workflow() -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("ingestion_node", ingestion_node)
    graph.add_node("template_attribute_node", template_attribute_node)

    graph.add_edge(START, "ingestion_node")
    graph.add_edge("ingestion_node", "template_attribute_node")
    graph.add_edge("template_attribute_node", END)

    return graph.compile()
