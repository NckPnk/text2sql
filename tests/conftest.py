from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from app.core.config import Settings
from app.core.models import SearchResult
from app.services.pipeline import Pipeline
from app.services.prompt_builder import PromptBuilder
from app.services.schema_retrieval import SchemaRetrievalService
from app.services.sql_executor import SQLExecutor
from app.services.sql_generator import SQLGenerator
from app.services.sql_validator import SQLValidator


@pytest.fixture
def sql_validator() -> SQLValidator:
    return SQLValidator(default_limit=100)


@pytest.fixture
def llm_client_mock() -> AsyncMock:
    client = AsyncMock()
    client.generate = AsyncMock()
    return client


@pytest.fixture
def embedding_client_mock() -> AsyncMock:
    client = AsyncMock()
    client.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return client


@pytest.fixture
def db_client_mock() -> AsyncMock:
    client = AsyncMock()
    client.execute = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client


@pytest.fixture
def chroma_client_mock() -> Mock:
    client = Mock()
    client.search = Mock(return_value=[])
    client.search_columns = Mock(return_value=[])
    return client


@pytest.fixture
def xdic_parser_mock() -> Mock:
    parser = Mock()
    parser.tables = {}

    ddls: dict[str, str] = {}
    descriptions: dict[str, str] = {}

    parser.get_create_table_sql = Mock(side_effect=lambda table_name: ddls.get(table_name))
    parser.get_table_context = Mock(
        side_effect=lambda table_name: {
            "description": descriptions.get(table_name, ""),
        }
    )
    parser._ddls = ddls
    parser._descriptions = descriptions
    return parser


@pytest.fixture
def make_search_result():
    def _make_search_result(
        table_name: str,
        score: float,
        metadata: dict | None = None,
    ) -> SearchResult:
        return SearchResult(
            table_name=table_name,
            score=score,
            metadata=metadata or {},
        )

    return _make_search_result


@pytest.fixture
def register_xdic_table(xdic_parser_mock: Mock):
    def _register_xdic_table(
        table_name: str,
        ddl: str,
        description: str = "",
        relations: list[dict[str, str]] | None = None,
    ) -> None:
        fields = {}
        for relation in relations or []:
            fields[relation["from_field"]] = SimpleNamespace(
                name=relation["from_field"],
                is_foreign_key=True,
                referenced_table=relation["to_table"],
            )

        xdic_parser_mock.tables[table_name] = SimpleNamespace(
            description=description,
            fields=fields,
        )
        xdic_parser_mock._ddls[table_name] = ddl
        xdic_parser_mock._descriptions[table_name] = description

    return _register_xdic_table


@pytest.fixture
def pipeline_settings(tmp_path) -> Settings:
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("Ты помощник по генерации SQL.", encoding="utf-8")
    return Settings(
        POSTGRES_HOST="localhost",
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
        PROMPT_PATH=str(prompt_path),
    )


@pytest.fixture
def prompt_builder(pipeline_settings: Settings) -> PromptBuilder:
    return PromptBuilder(pipeline_settings)


@pytest.fixture
def schema_retrieval_service(
    embedding_client_mock: AsyncMock,
    chroma_client_mock: Mock,
    xdic_parser_mock: Mock,
) -> SchemaRetrievalService:
    return SchemaRetrievalService(
        embedding_client=embedding_client_mock,
        vector_store=chroma_client_mock,
        xdic_parser=xdic_parser_mock,
    )


@pytest.fixture
def sql_generator(
    llm_client_mock: AsyncMock,
    sql_validator: SQLValidator,
) -> SQLGenerator:
    return SQLGenerator(llm_client=llm_client_mock, sql_validator=sql_validator)


@pytest.fixture
def sql_executor(db_client_mock: AsyncMock) -> SQLExecutor:
    return SQLExecutor(db_client=db_client_mock)


@pytest.fixture
def pipeline_factory(
    schema_retrieval_service: SchemaRetrievalService,
    prompt_builder: PromptBuilder,
    sql_generator: SQLGenerator,
    sql_validator: SQLValidator,
    sql_executor: SQLExecutor,
):
    def _pipeline_factory(max_retries: int = 1) -> Pipeline:
        return Pipeline(
            schema_retrieval=schema_retrieval_service,
            prompt_builder=prompt_builder,
            sql_generator=sql_generator,
            sql_validator=sql_validator,
            sql_executor=sql_executor,
            max_retries=max_retries,
        )

    return _pipeline_factory
