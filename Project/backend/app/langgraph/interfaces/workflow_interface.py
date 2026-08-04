"""
WorkflowInterface — abstract contract that every workflow implementation must satisfy.

Design intent
-------------
- The interface decouples the *what* (invoke a workflow, stream events, get the
  compiled graph) from the *how* (LangGraph, plain function chains, etc.).
- FastAPI endpoints, background workers, and tests depend only on this ABC,
  never on the concrete LangGraph adapter directly.
- The ``LangGraphAdapter`` in ``app/langgraph/adapters/langgraph_adapter.py``
  is the canonical implementation.

Usage example
-------------
    from app.langgraph.interfaces import WorkflowInterface
    from app.langgraph.adapters.langgraph_adapter import LangGraphAdapter
    from app.langgraph.workflows.document_workflow import build_document_workflow

    workflow: WorkflowInterface = LangGraphAdapter(build_document_workflow())
    result = await workflow.invoke(WorkflowInput(...))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class WorkflowInterface(ABC):
    """Abstract base class for all LangGraph (and future) workflow adapters."""

    # ---------------------------------------------------------------------- #
    # Core invocation                                                         #
    # ---------------------------------------------------------------------- #

    @abstractmethod
    async def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the full workflow synchronously and return the final state.

        Parameters
        ----------
        inputs:
            Initial state values to populate ``GraphState``.  The caller is
            responsible for validating inputs against ``WorkflowInput`` before
            passing them here.

        Returns
        -------
        dict
            The final ``GraphState`` after all nodes have executed.
        """
        ...

    # ---------------------------------------------------------------------- #
    # Streaming                                                               #
    # ---------------------------------------------------------------------- #

    @abstractmethod
    async def stream(self, inputs: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream intermediate node outputs as they become available.

        Yields one dict per node execution containing the partial state update
        produced by that node.  Useful for progress indicators and SSE endpoints.

        Parameters
        ----------
        inputs:
            Same as ``invoke``.

        Yields
        ------
        dict
            Partial state update from each completed node.
        """
        ...

    # ---------------------------------------------------------------------- #
    # Introspection                                                           #
    # ---------------------------------------------------------------------- #

    @abstractmethod
    def get_graph(self) -> Any:
        """Return the compiled underlying graph object.

        For LangGraph this is a ``CompiledStateGraph``.  Other implementations
        may return their own graph representation.  Useful for visualisation,
        debugging, and testing.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this workflow (e.g. 'document_workflow')."""
        ...
