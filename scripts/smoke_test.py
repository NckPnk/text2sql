from __future__ import annotations

import argparse
import json
import site
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import chromadb
import httpx
import ollama
import psycopg
from pydantic import ValidationError


def bootstrap_project_root() -> Path:
    """Make the project package importable for standalone script execution."""

    project_root = Path(__file__).resolve().parents[1]
    site.addsitedir(str(project_root))
    return project_root


PROJECT_ROOT = bootstrap_project_root()

from app.core.config import Settings
from app.infrastructure.xdic.parser import XdicParser


Status = Literal["ok", "warning", "error", "skipped"]


@dataclass
class CheckResult:
    key: str
    label: str
    status: Status
    message: str
    duration_ms: float


class SmokeTestRunner:
    def __init__(self, settings: Settings, full: bool, api_url: str):
        self.settings = settings
        self.full = full
        self.api_url = api_url.rstrip("/")
        self.results: list[CheckResult] = []
        self._ollama_available = False
        self._db_connected = False
        self._db_search_path: str | None = None
        self._db_connection: psycopg.Connection[Any] | None = None
        self._chroma_count: int | None = None
        self._xdic_count: int | None = None

    def run(self) -> int:
        try:
            self._check_ollama()
            self._check_postgres()
            self._check_chromadb()
            self._check_index_freshness()
            if self.full:
                self._check_e2e_query()
        finally:
            if self._db_connection is not None:
                self._db_connection.close()

        return 1 if any(result.status == "error" for result in self.results) else 0

    def emit_human_report(self) -> None:
        print("═══ Smoke Test ═══")
        print()

        for result in self.results:
            print(f"[{self._icon(result.status)}] {result.label:<22} {result.message}")

        ok_count = sum(result.status == "ok" for result in self.results)
        warning_count = sum(result.status == "warning" for result in self.results)
        error_count = sum(result.status == "error" for result in self.results)
        skipped_count = sum(result.status == "skipped" for result in self.results)

        print()
        summary = f"Итог: {ok_count}/{len(self.results)} ✅"
        if warning_count:
            summary += f", {warning_count} ⚠️"
        if error_count:
            summary += f", {error_count} ❌"
        if skipped_count:
            summary += f", {skipped_count} ⏭"
        print(summary)

    def emit_json_report(self) -> None:
        ok_count = sum(result.status == "ok" for result in self.results)
        warning_count = sum(result.status == "warning" for result in self.results)
        error_count = sum(result.status == "error" for result in self.results)
        skipped_count = sum(result.status == "skipped" for result in self.results)

        payload = {
            "overall_status": "error" if error_count else "ok",
            "summary": {
                "total": len(self.results),
                "ok": ok_count,
                "warning": warning_count,
                "error": error_count,
                "skipped": skipped_count,
            },
            "checks": [asdict(result) for result in self.results],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    def _check_ollama(self) -> None:
        started = time.perf_counter()
        try:
            client = ollama.Client(host=self.settings.ollama_base_url, timeout=5.0)
            response = client.list()
            models = response.get("models", [])
            model_names = {
                model.get("name", model.get("model", ""))
                for model in models
            }
            self._ollama_available = True
            self._add_result(
                "ollama_api",
                "Ollama API",
                "ok",
                "доступен",
                started,
            )

            self._record_model_check(
                started_at=None,
                key="ollama_llm_model",
                label="Модель LLM",
                model_name=self.settings.llm_model,
                available_models=model_names,
            )
            self._record_model_check(
                started_at=None,
                key="ollama_embed_model",
                label="Модель Embedding",
                model_name=self.settings.embed_model,
                available_models=model_names,
            )
        except Exception as exc:
            self._ollama_available = False
            self._add_result(
                "ollama_api",
                "Ollama API",
                "error",
                self._exc_message(exc, fallback=f"недоступен: {self.settings.ollama_base_url}"),
                started,
            )
            self._add_result(
                "ollama_llm_model",
                "Модель LLM",
                "error",
                "проверка невозможна: Ollama API недоступен",
            )
            self._add_result(
                "ollama_embed_model",
                "Модель Embedding",
                "error",
                "проверка невозможна: Ollama API недоступен",
            )

    def _record_model_check(
        self,
        *,
        started_at: float | None,
        key: str,
        label: str,
        model_name: str,
        available_models: set[str],
    ) -> None:
        started = started_at or time.perf_counter()
        status = "ok" if model_name in available_models else "error"
        message = (
            f"{model_name} загружена"
            if status == "ok"
            else f"{model_name} не найдена"
        )
        self._add_result(key, label, status, message, started)

    def _check_postgres(self) -> None:
        self._connect_postgres()
        self._check_search_path()
        self._check_select_one()
        self._check_accounts_table()

    def _connect_postgres(self) -> None:
        started = time.perf_counter()
        search_path = self._quoted_search_path()

        try:
            self._db_connection = psycopg.connect(
                host=self.settings.db_host,
                port=self.settings.db_port,
                dbname=self.settings.db_name,
                user=self.settings.db_user,
                password=self.settings.db_password.get_secret_value(),
                options=f"-c search_path={search_path}",
                connect_timeout=5,
            )
            self._db_connected = True
            self._add_result(
                "postgres_connection",
                "PostgreSQL",
                "ok",
                "подключение ОК",
                started,
            )
        except Exception as exc:
            self._db_connected = False
            self._db_connection = None
            self._add_result(
                "postgres_connection",
                "PostgreSQL",
                "error",
                self._exc_message(exc, fallback="подключение не удалось"),
                started,
            )

    def _check_search_path(self) -> None:
        started = time.perf_counter()
        if not self._db_connected or self._db_connection is None:
            self._add_result(
                "postgres_search_path",
                "search_path",
                "error",
                "проверка невозможна: PostgreSQL недоступен",
                started,
            )
            return

        try:
            with self._db_connection.cursor() as cursor:
                cursor.execute("SHOW search_path")
                value = cursor.fetchone()[0]
            self._db_search_path = value
            has_schema = self._search_path_contains_schema(value, self.settings.db_schema)
            status: Status = "ok" if has_schema else "error"
            message = value if has_schema else f"{value} — нет схемы {self.settings.db_schema}"
            self._add_result("postgres_search_path", "search_path", status, message, started)
        except Exception as exc:
            self._add_result(
                "postgres_search_path",
                "search_path",
                "error",
                self._exc_message(exc, fallback="не удалось прочитать"),
                started,
            )

    def _check_select_one(self) -> None:
        started = time.perf_counter()
        if not self._db_connected or self._db_connection is None:
            self._add_result(
                "postgres_select_1",
                "SELECT 1",
                "error",
                "проверка невозможна: PostgreSQL недоступен",
                started,
            )
            return

        try:
            with self._db_connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                value = cursor.fetchone()[0]
            status: Status = "ok" if value == 1 else "error"
            message = "простой запрос работает" if value == 1 else f"неожиданный ответ: {value}"
            self._add_result("postgres_select_1", "SELECT 1", status, message, started)
        except Exception as exc:
            self._add_result(
                "postgres_select_1",
                "SELECT 1",
                "error",
                self._exc_message(exc, fallback="не выполнился"),
                started,
            )

    def _check_accounts_table(self) -> None:
        started = time.perf_counter()
        if not self._db_connected or self._db_connection is None:
            self._add_result(
                "postgres_accounts_table",
                "Таблица тест",
                "error",
                "проверка невозможна: PostgreSQL недоступен",
                started,
            )
            return

        try:
            with self._db_connection.cursor() as cursor:
                cursor.execute('SELECT COUNT(*) FROM "Лицевые счета" LIMIT 1')
                row_count = cursor.fetchone()[0]
            self._add_result(
                "postgres_accounts_table",
                "Таблица тест",
                "ok",
                f'"Лицевые счета" доступна, строк: {row_count}',
                started,
            )
        except Exception as exc:
            self._add_result(
                "postgres_accounts_table",
                "Таблица тест",
                "error",
                self._exc_message(exc, fallback='таблица "Лицевые счета" недоступна'),
                started,
            )

    def _check_chromadb(self) -> None:
        started = time.perf_counter()
        try:
            client = chromadb.PersistentClient(path=str(self._resolve_path(self.settings.chroma_path)))
            collection = client.get_collection(self.settings.chroma_collection)
            count = collection.count()
            self._chroma_count = count

            if count > 0:
                message = f"коллекция найдена, {count} записей"
                status: Status = "ok"
            else:
                message = "коллекция найдена, но пуста"
                status = "error"

            self._add_result("chromadb_collection", "ChromaDB", status, message, started)
        except Exception as exc:
            self._chroma_count = None
            self._add_result(
                "chromadb_collection",
                "ChromaDB",
                "error",
                self._exc_message(exc, fallback="коллекция не найдена"),
                started,
            )

    def _check_index_freshness(self) -> None:
        started = time.perf_counter()
        try:
            parser = XdicParser(str(self._resolve_path(self.settings.xdic_path)))
            parser.parse()
            self._xdic_count = len(parser.tables)
        except Exception as exc:
            self._xdic_count = None
            self._add_result(
                "index_freshness",
                "Индекс актуальность",
                "error",
                self._exc_message(exc, fallback="не удалось прочитать .xdic"),
                started,
            )
            return

        if self._chroma_count is None:
            self._add_result(
                "index_freshness",
                "Индекс актуальность",
                "error",
                f"xdic: {self._xdic_count}, chroma: неизвестно — ChromaDB недоступна",
                started,
            )
            return

        if self._xdic_count == self._chroma_count:
            self._add_result(
                "index_freshness",
                "Индекс актуальность",
                "ok",
                f"xdic: {self._xdic_count}, chroma: {self._chroma_count}",
                started,
            )
            return

        self._add_result(
            "index_freshness",
            "Индекс актуальность",
            "warning",
            (
                f"xdic: {self._xdic_count}, chroma: {self._chroma_count} — "
                "РАССИНХРОН, запустите scripts/index_schema.py"
            ),
            started,
        )

    def _check_e2e_query(self) -> None:
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.api_url}/api/query",
                    json={"question": "Сколько лицевых счетов в базе?"},
                )
                response.raise_for_status()
                payload = response.json()

            success = bool(payload.get("success"))
            total_ms = ((payload.get("timings") or {}).get("total_ms")) or 0
            status: Status = "ok" if success else "error"
            seconds = total_ms / 1000 if total_ms else time.perf_counter() - started
            message = f"success={str(success).lower()}, {seconds:.1f}s ({self.api_url})"
            self._add_result("e2e_query", "E2E запрос", status, message, started)
        except Exception as exc:
            self._add_result(
                "e2e_query",
                "E2E запрос",
                "error",
                self._exc_message(exc, fallback=f"запрос к {self.api_url}/api/query не выполнен"),
                started,
            )

    def _quoted_search_path(self) -> str:
        schema = self.settings.db_schema.replace('"', '""')
        return f'"{schema}",public'

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _search_path_contains_schema(self, search_path: str | None, schema: str | None) -> bool:
        if not search_path or not schema:
            return False

        normalized_parts = {
            part.strip().strip('"').lower()
            for part in search_path.split(",")
        }
        return schema.lower() in normalized_parts

    def _add_result(
        self,
        key: str,
        label: str,
        status: Status,
        message: str,
        started_at: float | None = None,
    ) -> None:
        duration_ms = (
            round((time.perf_counter() - started_at) * 1000, 1)
            if started_at is not None
            else 0.0
        )
        self.results.append(
            CheckResult(
                key=key,
                label=label,
                status=status,
                message=message,
                duration_ms=duration_ms,
            )
        )

    def _exc_message(self, exc: Exception, fallback: str) -> str:
        if isinstance(exc, httpx.ConnectError):
            return fallback
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text.strip()
            return f"HTTP {exc.response.status_code}" + (f": {body}" if body else "")

        text = str(exc).strip()
        return text or fallback

    @staticmethod
    def _icon(status: Status) -> str:
        return {
            "ok": "✅",
            "warning": "⚠️",
            "error": "❌",
            "skipped": "⏭",
        }[status]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast post-deploy smoke test.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run one end-to-end request through the HTTP API.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human report.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL for the API used with --full.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = Settings()
    except ValidationError as exc:
        payload = {
            "overall_status": "error",
            "summary": {
                "total": 1,
                "ok": 0,
                "warning": 0,
                "error": 1,
                "skipped": 0,
            },
            "checks": [
                {
                    "key": "config",
                    "label": "Конфигурация",
                    "status": "error",
                    "message": str(exc),
                    "duration_ms": 0.0,
                }
            ],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("═══ Smoke Test ═══")
            print()
            print(f"[❌] {'Конфигурация':<22} некорректные или отсутствующие переменные окружения")
            print()
            print("Итог: 0/1 ✅, 1 ❌")
        return 1

    runner = SmokeTestRunner(settings=settings, full=args.full, api_url=args.api_url)
    exit_code = runner.run()

    if args.json:
        runner.emit_json_report()
    else:
        runner.emit_human_report()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
