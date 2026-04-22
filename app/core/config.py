"""Application configuration."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    db_host: str = Field(validation_alias="POSTGRES_HOST")
    db_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    db_name: str = Field(validation_alias="POSTGRES_DB")
    db_user: str = Field(validation_alias="POSTGRES_USER")
    db_password: SecretStr = Field(validation_alias="POSTGRES_PASSWORD")
    db_schema: str = Field(default="stack", validation_alias="POSTGRES_SCHEMA")

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    llm_model: str = Field(
        default="qwen3-coder:30b",
        validation_alias="OLLAMA_LLM_MODEL",
    )
    embed_model: str = Field(
        default="qwen3-embedding:8b",
        validation_alias="OLLAMA_EMBED_MODEL",
    )

    chroma_path: str = Field(
        default="./data/chroma",
        validation_alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection: str = Field(
        default="table_schemas",
        validation_alias="CHROMA_COLLECTION",
    )
    chroma_column_collection: str = Field(
        default="column_schemas",
        validation_alias="CHROMA_COLUMN_COLLECTION",
    )

    xdic_path: str = Field(
        default="./data/xdic/main.xdic",
        validation_alias="XDIC_PATH",
    )
    prompt_path: str = Field(
        default="./data/prompts/system_prompt_v1.txt",
        validation_alias="PROMPT_PATH",
    )

    llm_timeout: int = Field(default=120, validation_alias="LLM_TIMEOUT")
    embed_timeout: int = Field(default=30, validation_alias="EMBED_TIMEOUT")
    sql_timeout: int = Field(default=30, validation_alias="SQL_TIMEOUT")
    sql_max_rows: int = Field(default=500, validation_alias="SQL_MAX_ROWS")
    sql_default_limit: int = Field(
        default=100,
        validation_alias="SQL_DEFAULT_LIMIT",
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def db_url(self) -> str:
        """Return SQLAlchemy asyncpg connection URL."""

        password = self.db_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.db_user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def db_url_sync(self) -> str:
        """Return synchronous PostgreSQL connection URL."""

        password = self.db_password.get_secret_value()
        return (
            f"postgresql://{self.db_user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


__all__ = ["Settings"]
