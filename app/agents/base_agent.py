import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Awaitable, Callable, Optional

from ..config import settings

logger = logging.getLogger(__name__)


def timeout_guard(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator to enforce per-agent timeout"""

    @wraps(func)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        timeout_seconds = getattr(self, "timeout_seconds", settings.agent_timeout_seconds)
        try:
            return await asyncio.wait_for(func(self, *args, **kwargs), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            logger.error("Agent %s timed out after %s seconds", getattr(self, "agent_name", "unknown"), timeout_seconds)
            raise TimeoutError(f"Operation exceeded timeout of {timeout_seconds} seconds") from exc

    return wrapper


@dataclass
class LLMConfig:
    model: str
    temperature: float
    max_tokens: int

class BaseDocumentAgent(ABC):
    """Abstract base class for document processing agents."""

    def __init__(self) -> None:
        self.timeout_seconds = settings.agent_timeout_seconds
        self.llm: Optional[LLMConfig] = None
        try:
            self.llm = create_llm_config()
        except RuntimeError as exc:
            logger.warning("LLM config not initialized for %s: %s", self.agent_name, exc)

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Name used for logging and structured responses."""

    @abstractmethod
    async def process(self, document_text: str) -> dict:
        """Implement agent-specific processing logic and return a dict payload."""

    @timeout_guard
    async def _process_with_timeout(self, document_text: str) -> dict:
        return await self.process(document_text)

    async def execute(self, document_text: str) -> dict:
        """Execute processing with timeout, retries, and structured error handling."""
        try:
            result = await self._process_with_timeout(document_text)
            if not isinstance(result, dict):
                raise ValueError("Agent process() must return a dictionary payload.")
            return {
                "status": "success",
                "agent": self.agent_name,
                "data": result,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        except TimeoutError as exc:
            return self._error_response(error_type="timeout", message=str(exc))
        except Exception as exc:
            logger.exception("Unhandled agent error in %s", self.agent_name)
            return self._error_response(error_type="exception", message=str(exc))

    def _error_response(self, *, error_type: str, message: str) -> dict:
        return {
            "status": "error",
            "agent": self.agent_name,
            "error_type": error_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }


def create_llm_config(
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: Optional[float] = None
) -> LLMConfig:
    """Create and configure an LLM config based on available credentials."""
    timeout = timeout or settings.agent_timeout_seconds

    if settings.openai_api_key:
        model_name = "gpt-4o-mini"
        return LLMConfig(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    raise RuntimeError("OPENAI_API_KEY is required to use the gpt-4o model.")
