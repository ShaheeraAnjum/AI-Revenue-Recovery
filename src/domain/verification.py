"""Verification and gateway execution response domain models."""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from src.domain.actions import ActionType


class VerificationStatus(str, Enum):
    """Execution verification status enum."""
    SUCCESS = "SUCCESS"
    DECLINE = "DECLINE"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"
    RECONCILIATION = "RECONCILIATION"


class GatewayExecutionResult(BaseModel):
    """Verification result returned by the idempotent action executor."""
    execution_id: str
    case_id: str
    decision_id: str
    action: ActionType
    attempt: int
    idempotency_key: str
    status: VerificationStatus
    gateway_reference: Optional[str] = None
    response_code: Optional[str] = None
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
