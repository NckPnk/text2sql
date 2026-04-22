"""Core Pydantic models for the Text2SQL pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    """Incoming natural-language query from a user."""

    model_config = ConfigDict(frozen=True)

    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be empty")
        return question


class TableContext(BaseModel):
    """Context about a retrieved database table."""

    model_config = ConfigDict(frozen=True)

    name: str
    ddl: str
    description: str = ""
    relevance_score: float = 0.0
    relations: list[str] = Field(default_factory=list)
    matched_columns: list[str] = Field(default_factory=list)
    score_components: dict[str, float] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """SQL validation outcome."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    original_sql: str
    fixed_sql: str | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    """LLM SQL generation result."""

    model_config = ConfigDict(frozen=True)

    raw_response: str
    extracted_sql: str | None = None
    explanation: str | None = None


class SQLResult(BaseModel):
    """Executed SQL result set."""

    model_config = ConfigDict(frozen=True)

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    execution_time_ms: float


class PipelineTimings(BaseModel):
    """Timings for each pipeline stage."""

    model_config = ConfigDict(frozen=True)

    schema_retrieval_ms: float = 0.0
    prompt_build_ms: float = 0.0
    llm_generation_ms: float = 0.0
    validation_ms: float = 0.0
    execution_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    """Full response returned by the Text2SQL system."""

    model_config = ConfigDict()

    question: str
    sql: str | None = None
    result: SQLResult | None = None
    tables_used: list[str] = Field(default_factory=list)
    timings: PipelineTimings = Field(default_factory=PipelineTimings)
    error: str | None = None
    success: bool = False


class QueryProgressEvent(BaseModel):
    """Pipeline progress event for streaming responses."""

    model_config = ConfigDict()

    phase: Literal[
        "retrieval",
        "generation",
        "validation",
        "execution",
        "done",
        "error",
    ]
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    response: QueryResponse | None = None


class SearchResult(BaseModel):
    """A vector store search hit."""

    model_config = ConfigDict(frozen=True)

    table_name: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ColumnSearchResult(BaseModel):
    """A vector store column-level search hit."""

    model_config = ConfigDict(frozen=True)

    table_name: str
    column_name: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Health endpoint payload."""

    model_config = ConfigDict()

    status: str = "ok"
    ollama_available: bool = False
    db_connected: bool = False
    chroma_tables_count: int = 0
    search_path: str | None = None
    schema_ok: bool = False


__all__ = [
    "GenerationResult",
    "HealthStatus",
    "PipelineTimings",
    "QueryProgressEvent",
    "QueryRequest",
    "QueryResponse",
    "SQLResult",
    "SearchResult",
    "ColumnSearchResult",
    "TableContext",
    "ValidationResult",
]
