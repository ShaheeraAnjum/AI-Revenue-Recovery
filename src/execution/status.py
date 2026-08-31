"""Execution outcome states and error codes for idempotent action execution."""
from enum import Enum


class ExecutionStatus(str, Enum):
    """Outcome status of an individual action execution attempt."""
    SUCCESS = "SUCCESS"
    DECLINE = "DECLINE"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"
    REJECTED = "REJECTED"
    DUPLICATE = "DUPLICATE"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
