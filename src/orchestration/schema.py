"""Data schemas for end-to-end revenue recovery orchestration."""
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import CaseState
from src.engine.decision import DecisionResult
from src.execution.executor import ExecutionResult
from src.verification.observation import ObservationRecord
from src.messaging.schema import MessageAuditRecord, CandidateMessage, ValidationResult


class OrchestrationCycleResult(BaseModel):
    """Complete auditable output of a single recovery decision-execution-observation-messaging cycle."""
    cycle_id: str
    case_id: str
    decision_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    selected_action: Optional[ActionType] = None
    decision_result: Optional[DecisionResult] = None
    execution_result: Optional[ExecutionResult] = None
    observation_record: Optional[ObservationRecord] = None
    delivered_message: Optional[CandidateMessage] = None
    message_validation: Optional[ValidationResult] = None
    message_audit: Optional[MessageAuditRecord] = None
    case_state: CaseState
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
