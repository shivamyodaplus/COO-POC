"""
LangGraphAdapter — concrete implementation of ``WorkflowInterface`` using LangGraph.

This is the only place in the codebase that knows about LangGraph internals
(``StateGraph``, ``CompiledStateGraph``, checkpointers, etc.).  All FastAPI
routes and services interact with this class through the ``WorkflowInterface``
ABC only.

Typical wiring
--------------
    from app.langgraph.adapters.langgraph_adapter import LangGraphAdapter
    from app.langgraph.workflows.document_workflow import build_document_workflow

    adapter = LangGraphAdapter(
        compiled_graph=build_document_workflow(),
        name="document_workflow",
    )

    # In an async FastAPI endpoint:
    result = await adapter.invoke({"raw_bytes": ..., "filename": ...})

    # Streaming (SSE / WebSocket):
    async for update in adapter.stream({"raw_bytes": ..., "filename": ...}):
        yield update
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from langgraph.graph.state import CompiledStateGraph

from app.langgraph.interfaces.workflow_interface import WorkflowInterface

logger = logging.getLogger(__name__)


class LangGraphAdapter(WorkflowInterface):
    """Wraps a compiled LangGraph ``StateGraph`` behind ``WorkflowInterface``."""

    def __init__(self, compiled_graph: CompiledStateGraph, name: str) -> None:
        """
        Parameters
        ----------
        compiled_graph:
            A ``StateGraph`` that has already been compiled via ``.compile()``.
        name:
            Human-readable workflow identifier (used in logs and ``get_name()``).
        """
        self._graph = compiled_graph
        self._name = name

    # ---------------------------------------------------------------------- #
    # WorkflowInterface implementation                                        #
    # ---------------------------------------------------------------------- #

    async def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the full graph and return the terminal state.

        LangGraph's synchronous ``invoke`` is offloaded to a thread so that
        the async event loop is never blocked.
        """
        logger.info("LangGraphAdapter[%s]: invoke called", self._name)
        result: dict[str, Any] = await asyncio.to_thread(self._graph.invoke, inputs)
        logger.info(
            "LangGraphAdapter[%s]: invoke complete — current_step=%s errors=%s",
            self._name,
            result.get("current_step"),
            result.get("errors"),
        )
        return result

    async def stream(self, inputs: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Yield partial state updates from each node as they complete.

        LangGraph's synchronous generator is consumed in a thread via an
        async queue so the event loop stays non-blocking.
        """
        logger.info("LangGraphAdapter[%s]: stream called", self._name)
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def _produce() -> None:
            try:
                for chunk in self._graph.stream(inputs):
                    # ``asyncio.Queue.put_nowait`` is not thread-safe; use the
                    # loop-safe call_soon_threadsafe approach via run_coroutine.
                    asyncio.get_event_loop().call_soon_threadsafe(
                        queue.put_nowait, chunk
                    )
            finally:
                asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, None)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _produce)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def get_graph(self) -> CompiledStateGraph:
        """Return the underlying compiled LangGraph object."""
        return self._graph

    def get_name(self) -> str:
        """Return the workflow name supplied at construction time."""
        return self._name
