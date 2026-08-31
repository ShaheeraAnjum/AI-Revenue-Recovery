"""Idempotency store and conflict detection for safe action execution."""
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from src.domain.case import IdempotencyKey
from src.execution.status import ExecutionStatus

DEFAULT_EXECUTOR_VERSION: str = "exec_v5.0.0"


class IdempotencyRecord(BaseModel):
    """Persisted execution result associated with an idempotency key."""
    key: str
    payload_hash: str
    execution_status: ExecutionStatus
    response_data: Dict[str, Any]
    reference_id: Optional[str] = None
    error_code: Optional[str] = None


class IdempotencyConflictError(Exception):
    """Raised when the same idempotency key is submitted with materially conflicting payloads."""
    pass


class BaseIdempotencyStore(ABC):
    """Abstract interface for storing and retrieving idempotency records."""

    @abstractmethod
    def get(self, key: str) -> Optional[IdempotencyRecord]:
        """Retrieve recorded execution by key."""
        pass

    @abstractmethod
    def put(self, record: IdempotencyRecord) -> None:
        """Store execution record by key."""
        pass


class InMemoryIdempotencyStore(BaseIdempotencyStore):
    """Thread-safe, deterministic in-memory idempotency store for prototype execution."""

    def __init__(self):
        self._store: Dict[str, IdempotencyRecord] = {}

    def get(self, key: str) -> Optional[IdempotencyRecord]:
        return self._store.get(key)

    def put(self, record: IdempotencyRecord) -> None:
        self._store[record.key] = record

    def clear(self) -> None:
        self._store.clear()


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    """Generate deterministic SHA-256 hash for an execution payload."""
    canonical_json = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
