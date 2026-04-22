from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import PromptBuildError
from app.core.models import TableContext

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build prompts for SQL generation."""

    def __init__(self, settings: Settings):
        path = Path(settings.prompt_path)
        try:
            self._system_prompt = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise PromptBuildError(f"Файл промта не найден: {path}") from exc

        if not self._system_prompt.strip():
            raise PromptBuildError("Файл промта пуст")

        self._max_context_chars = 15000
        logger.info(
            "Системный промт загружен, длина: %s символов",
            len(self._system_prompt),
        )

    def build(self, question: str, tables: list[TableContext]) -> tuple[str, str]:
        if not tables:
            raise PromptBuildError("Нет таблиц для контекста")

        sorted_tables = sorted(
            tables,
            key=lambda table: table.relevance_score,
            reverse=True,
        )

        ddl_blocks: list[str] = []
        relations: list[str] = []
        seen_relations: set[str] = set()
        context_chars = 0
        added_tables = 0

        for table in sorted_tables:
            description = table.description.strip()
            matched_columns_block = ""
            if table.matched_columns:
                matched_columns_block = (
                    "Matched columns: "
                    + ", ".join(table.matched_columns[:12])
                )
            table_block = (
                f"--- Таблица: {table.name} ---\n"
                f"{description}\n\n"
                f"{matched_columns_block}\n\n"
                f"{table.ddl.strip()}"
            )
            table_block_size = len(table_block)
            if context_chars + table_block_size > self._max_context_chars:
                logger.warning(
                    "Обрезано: добавлено %s из %s таблиц",
                    added_tables,
                    len(sorted_tables),
                )
                break

            ddl_blocks.append(table_block)
            context_chars += table_block_size
            added_tables += 1

            for relation in table.relations:
                if relation in seen_relations:
                    continue
                seen_relations.add(relation)
                relations.append(relation)

        ddl_section = "\n\n".join(ddl_blocks)

        relations_section = ""
        if relations:
            relations_section = (
                "--- Связи между таблицами ---\n"
                + "\n".join(relations)
            )

        prompt_parts = [
            "══════════════════════════════",
            "СХЕМА БАЗЫ ДАННЫХ",
            "══════════════════════════════",
            "",
            ddl_section,
        ]
        if relations_section:
            prompt_parts.extend(["", relations_section])
        prompt_parts.extend(
            [
                "",
                "══════════════════════════════",
                "ВОПРОС ПОЛЬЗОВАТЕЛЯ",
                "══════════════════════════════",
                "",
                question,
            ]
        )
        user_prompt = "\n".join(prompt_parts)

        logger.debug("Количество таблиц в контексте: %s", added_tables)
        logger.debug("Длина user_prompt: %s символов", len(user_prompt))
        logger.debug(
            "Общая длина system + user: %s символов",
            len(self._system_prompt) + len(user_prompt),
        )

        return self._system_prompt, user_prompt

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 3
