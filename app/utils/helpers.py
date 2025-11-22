import uuid
import re
from datetime import datetime
from typing import Dict
from ..models.schemas import StatusEnum

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

