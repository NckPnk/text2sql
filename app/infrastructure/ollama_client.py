"""Client for Ollama LLM and embedding APIs via official SDK."""

from __future__ import annotations

import logging
import time
from typing import Any

import asyncio
import ollama

from app.core.config import Settings
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async Ollama client implementing LLM and embedding operations."""

    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url
        self._llm_model = settings.llm_model
        self._embed_model = settings.embed_model
        self._llm_timeout = settings.llm_timeout
        self._embed_timeout = settings.embed_timeout

        self._llm_client = ollama.AsyncClient(
            host=self._base_url,
            timeout=self._llm_timeout,
        )
        self._embed_client = ollama.AsyncClient(
            host=self._base_url,
            timeout=self._embed_timeout,
        )

    async def generate(self, prompt: str, system: str) -> str:
        """Generate text from Ollama chat API."""

        logger.debug(
            "Generating text with prompt_length=%s system_length=%s",
            len(prompt),
            len(system),
        )
        started_at = time.perf_counter()

        try:
            response = await self._chat_with_retry(
                client=self._llm_client,
                max_retries=2,
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 4096,
                },
            )
            content = response["message"]["content"]
            if not content:
                raise LLMError("LLM вернула пустой ответ")
        except TimeoutError as exc:
            message = f"Ollama таймаут ({self._llm_timeout}с)"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except ollama.ResponseError as exc:
            message = self._build_llm_http_error(exc, self._llm_model)
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except ConnectionError as exc:
            message = f"Ollama недоступна: {self._base_url}"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except LLMError as exc:
            logger.error(str(exc), exc_info=exc)
            raise
        except (KeyError, TypeError, ValueError) as exc:
            message = "Некорректный ответ от Ollama LLM"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc

        elapsed = time.perf_counter() - started_at
        logger.info(
            "Ollama generate completed model=%s duration=%.3fs content_length=%s",
            self._llm_model,
            elapsed,
            len(content),
        )
        return content

    async def embed(self, text: str) -> list[float]:
        """Fetch embedding vector from Ollama embed API."""

        logger.debug("Generating embedding for text_length=%s", len(text))
        try:
            response = await self._embed_with_retry(
                client=self._embed_client,
                max_retries=2,
                model=self._embed_model,
                input=text,
            )
            embedding = response["embeddings"][0]
            if not embedding:
                raise LLMError("Embedding вернула пустой вектор")
        except TimeoutError as exc:
            message = f"Ollama embedding таймаут ({self._embed_timeout}с)"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except ollama.ResponseError as exc:
            message = self._build_embedding_http_error(exc, self._embed_model)
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except ConnectionError as exc:
            message = f"Ollama embedding недоступна: {self._base_url}"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except LLMError as exc:
            logger.error(str(exc), exc_info=exc)
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            message = "Некорректный ответ от Ollama embedding"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc

        logger.debug("Received embedding vector with dimension=%s", len(embedding))
        return embedding

    async def is_available(self) -> bool:
        """Check whether Ollama responds to a basic availability probe."""

        try:
            await self._llm_client.list()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        """SDK clients are stateless, nothing to close."""

    async def _chat_with_retry(
        self,
        client: ollama.AsyncClient,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send chat requests with retries for Ollama 5xx responses."""

        return await self._request_with_retry(
            requester=client.chat,
            max_retries=max_retries,
            include_stream=True,
            **kwargs,
        )

    async def _embed_with_retry(
        self,
        client: ollama.AsyncClient,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send embedding requests with retries for Ollama 5xx responses."""

        return await self._request_with_retry(
            requester=client.embed,
            max_retries=max_retries,
            include_stream=False,
            **kwargs,
        )

    async def _request_with_retry(
        self,
        requester: Any,
        max_retries: int = 2,
        include_stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send request with retries for Ollama 5xx responses."""

        last_status_code: int | None = None

        for attempt in range(1, max_retries + 2):
            try:
                if include_stream:
                    response = await requester(stream=False, **kwargs)
                else:
                    response = await requester(**kwargs)
                return response
            except ollama.ResponseError as exc:
                last_status_code = getattr(exc, "status_code", None)
                if not last_status_code or last_status_code < 500:
                    raise

            if attempt > max_retries:
                break

            await asyncio.sleep(attempt)

        raise LLMError(
            "Ollama вернула серверную ошибку "
            f"({last_status_code}) после {max_retries + 1} попыток"
        )

    @staticmethod
    def _build_llm_http_error(exc: ollama.ResponseError, model: str) -> str:
        if getattr(exc, "status_code", None) == 404:
            return f"Модель {model} не найдена в Ollama"
        return f"Ollama LLM HTTP ошибка: {getattr(exc, 'status_code', 'unknown')}"

    @staticmethod
    def _build_embedding_http_error(exc: ollama.ResponseError, model: str) -> str:
        if getattr(exc, "status_code", None) == 404:
            return f"Модель {model} не найдена в Ollama"
        return f"Ollama embedding HTTP ошибка: {getattr(exc, 'status_code', 'unknown')}"
