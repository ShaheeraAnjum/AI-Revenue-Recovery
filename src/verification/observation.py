"""Observation records and window tracking for provisional execution results."""
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import CaseState
from src.domain.verification import VerificationStatus
from src.execution.status import ExecutionStatus
from src.execution.executor import ExecutionResult

DEFAULT_OBSERVATION_VERSION: str = "obs_v5.0.0"


class ReconciliationStatus(str, Enum):
    """Reconciliation lifecycle status."""
    PENDING = "PENDING"
    SETTLED = "SETTLED"
    REFUNDED = "REFUNDED"
    CHARGEBACK = "CHARGEBACK"
    DISPUTED = "DISPUTED"
    FAILED = "FAILED"


class ObservationConfig(BaseModel):
    """Versioned configuration for observation window duration."""
    observation_version: str = DEFAULT_OBSERVATION_VERSION
    default_window_seconds: int = Field(default=3600, ge=0, description="Observation holding window in seconds")


class ObservationRecord(BaseModel):
    """Persisted observation tracking record bridging provisional execution and reconciliation."""
    observation_id: str
    case_id: str
    decision_id: str
    action: ActionType
    attempt: int
    idempotency_key: str
    execution_status: ExecutionStatus
    execution_timestamp: datetime
    observation_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    observation_window_seconds: int
    observation_deadline: datetime
    is_provisional: bool = True
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN
    reconciliation_status: ReconciliationStatus = ReconciliationStatus.PENDING
    final_outcome: Optional[CaseState] = None
    reconciliation_fingerprint: Optional[str] = None
    observation_version: str = DEFAULT_OBSERVATION_VERSION
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_window_elapsed(self, as_of_time: Optional[datetime] = None) -> bool:
        """Check if observation window has elapsed as of specified timestamp."""
        check_time = as_of_time or datetime.now(timezone.utc)
        return check_time >= self.observation_deadline

    @property
    def is_finalized(self) -> bool:
        """True if observation has reached terminal outcome."""
        return self.final_outcome in {CaseState.RESOLVED_RECOVERED, CaseState.RESOLVED_UNRECOVERABLE}
