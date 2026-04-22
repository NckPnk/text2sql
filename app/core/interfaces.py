"""Protocol interfaces for infrastructure and service adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core.models import ColumnSearchResult, SearchResult


@runtime_checkable
class LLMClient(Protocol):
    """Contract for text generation clients."""

    async def generate(self, prompt: str, system: str) -> str:
        """Generate text where `prompt` is the user message and `system` is the system message."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Contract for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a text."""


@runtime_checkable
class DatabaseClient(Protocol):
    """Contract for database access implementations."""

    async def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute SQL and return rows as `list[dict]`."""

    async def connect(self) -> None:
        """Initialize the connection or connection pool."""

    async def disconnect(self) -> None:
        """Close the connection or connection pool."""


@runtime_checkable
class VectorStore(Protocol):
    """Contract for vector similarity search backends."""

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
    ) -> list[SearchResult]:
        """Search nearest items for the given query embedding."""

    def search_columns(
        self,
        query_embedding: list[float],
        n_results: int = 20,
    ) -> list[ColumnSearchResult]:
        """Search nearest columns for the given query embedding."""


__all__ = ["DatabaseClient", "EmbeddingClient", "LLMClient", "VectorStore"]
