"""Thread-safe atomic idempotency store and conflict detection for safe action execution."""
import hashlib
import json
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple, Callable
from pydantic import BaseModel, Field
from src.domain.case import IdempotencyKey
from src.execution.status import ExecutionStatus

DEFAULT_EXECUTOR_VERSION: str = "exec_v5.0.0"


class IdempotencyRecord(BaseModel):
    """Persisted execution result associated with an idempotency key."""
    key: str
    payload_hash: str
    execution_status: ExecutionStatus
    response_data: Dict[str, Any] = Field(default_factory=dict)
    reference_id: Optional[str] = None
    error_code: Optional[str] = None
    adapter_version: str = "adapter_v5.0.0"


class IdempotencyConflictError(Exception):
    """Raised when the same idempotency key is submitted with materially conflicting payloads."""
    pass


class BaseIdempotencyStore(ABC):
    """Abstract interface for storing and atomically executing under idempotency keys."""

    @abstractmethod
    def get(self, key: str) -> Optional[IdempotencyRecord]:
        """Retrieve recorded execution by key."""
        pass

    @abstractmethod
    def execute_once(
        self,
        key: str,
        payload_hash: str,
        execution_fn: Callable[[], Any],
    ) -> Tuple[IdempotencyRecord, bool]:
        """Atomically execute the callback once per key.
        Returns: (record, is_duplicate)
        """
        pass


class InMemoryIdempotencyStore(BaseIdempotencyStore):
    """Thread-safe, atomic in-memory idempotency store with per-key locking."""

    def __init__(self):
        self._store: Dict[str, IdempotencyRecord] = {}
        self._global_lock = threading.Lock()
        self._key_locks: Dict[str, threading.Lock] = {}

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._global_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def get(self, key: str) -> Optional[IdempotencyRecord]:
        with self._global_lock:
            return self._store.get(key)

    def execute_once(
        self,
        key: str,
        payload_hash: str,
        execution_fn: Callable[[], Any],
    ) -> Tuple[IdempotencyRecord, bool]:
        """Atomically check, claim, execute, and store side effect under per-key lock.
        Prevents concurrent callers with the same key from duplicating external executions.
        """
        key_lock = self._get_key_lock(key)
        with key_lock:
            # Check existing record
            with self._global_lock:
                existing = self._store.get(key)

            if existing is not None:
                # Validate payload identity
                if existing.payload_hash != payload_hash:
                    raise IdempotencyConflictError(
                        f"Idempotency conflict: key '{key}' previously executed with different payload hash"
                    )
                return existing, True

            # First caller executes the side-effect callback
            adapter_resp = execution_fn()

            record = IdempotencyRecord(
                key=key,
                payload_hash=payload_hash,
                execution_status=adapter_resp.status,
                response_data=adapter_resp.details,
                reference_id=adapter_resp.reference_id,
                error_code=adapter_resp.error_code,
                adapter_version=adapter_resp.adapter_version,
            )

            with self._global_lock:
                self._store[key] = record

            return record, False

    def clear(self) -> None:
        with self._global_lock:
            self._store.clear()
            self._key_locks.clear()


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    """Generate deterministic SHA-256 hash for an execution payload."""
    canonical_json = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
