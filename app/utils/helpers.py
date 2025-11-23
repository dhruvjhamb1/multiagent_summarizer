import uuid
from datetime import datetime
from typing import Dict, Any, Callable
import random
import asyncio
from ..models.schemas import StatusEnum
import logging

logger = logging.getLogger(__name__)

def generate_job_id() -> str:
    """
    Generate a unique job ID with 'job_' prefix.

    Returns:
        str: Unique job identifier
    """
    return f"job_{uuid.uuid4()}"

def generate_document_id() -> str:
    """
    Generate a unique document ID with 'doc_' prefix.

    Returns:
        str: Unique document identifier
    """
    return f"doc_{uuid.uuid4()}"

def calculate_progress(agents_status: Dict[str, StatusEnum]) -> float:
    """
    Calculate completion progress percentage based on agent statuses.

    Args:
        agents_status: Dictionary mapping agent names to their status

    Returns:
        int: Progress percentage (0-100)
    """
    if not agents_status:
        return 0.0

    completed_count = sum(1 for status in agents_status.values() if status == StatusEnum.COMPLETED)
    total_count = len(agents_status)
    return round((completed_count / total_count) * 100, 2)

def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        str: ISO formatted timestamp string
    """
    return datetime.now().isoformat()


async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 10.0,
    multiplier: float = 2.0,
    jitter: float = 0.1,
    retry_exceptions: tuple = (Exception,),
    # Optional per-call timeout in seconds. If provided, each attempt will be wrapped
    # in asyncio.wait_for(..., timeout=call_timeout) so stuck requests don't block retries.
    call_timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Run an async callable with exponential backoff and jitter.

    Args:
        func: Async callable to execute.
        max_attempts: Maximum number of attempts before giving up.
        initial_delay: Base delay in seconds.
        max_delay: Maximum delay cap in seconds.
        multiplier: Backoff multiplier.
        jitter: Fractional jitter to add/subtract from computed delay.
        retry_exceptions: Tuple of exception classes to treat as retryable.
        kwargs: Keyword args forwarded to `func`.

    Returns:
        The result of the successful `func` call.

    Raises:
        The last exception raised by `func` if max attempts are exhausted.
    """

    attempt = 0
    while True:
        attempt += 1
        try:
            if call_timeout:
                # Enforce a per-call timeout to avoid indefinitely hanging calls.
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=call_timeout)
            else:
                result = await func(*args, **kwargs)
            logger.debug("retry_with_backoff: attempt %s succeeded", attempt)
            return result
        except retry_exceptions as exc:
            logger.warning(
                "retry_with_backoff: attempt %s failed with %s: %s",
                attempt,
                type(exc).__name__,
                exc,
            )
            if attempt >= max_attempts:
                raise
            # compute backoff with jitter
            delay = min(initial_delay * (multiplier ** (attempt - 1)), max_delay)
            jitter_amount = delay * jitter * (random.random() * 2 - 1)
            sleep_for = max(0.0, delay + jitter_amount)
            await asyncio.sleep(sleep_for)


