"""Idempotent action executor dispatching compliance-approved decisions."""
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile, IdempotencyKey
from src.execution.status import ExecutionStatus
from src.execution.idempotency import (
    BaseIdempotencyStore,
    InMemoryIdempotencyStore,
    IdempotencyRecord,
    IdempotencyConflictError,
    compute_payload_hash,
    DEFAULT_EXECUTOR_VERSION,
)
from src.execution.adapters import (
    BaseActionAdapter,
    PaymentRetryAdapter,
    PaymentUpdateAdapter,
    ReminderAdapter,
    WaitAdapter,
    EscalateAdapter,
    StopAdapter,
    DEFAULT_ADAPTER_VERSION,
)


class ExecutionResult(BaseModel):
    """Auditable result of executing an action with exact idempotency tracking.
    IMPORTANT: Execution SUCCESS is purely provisional and does NOT mark case as RESOLVED_RECOVERED.
    """
    case_id: str
    decision_id: str
    action: ActionType
    attempt: int
    idempotency_key: str
    execution_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_status: ExecutionStatus
    executor_version: str = DEFAULT_EXECUTOR_VERSION
    adapter_version: str = DEFAULT_ADAPTER_VERSION
    reference_id: Optional[str] = None
    error_code: Optional[str] = None
    is_duplicate: bool = False
    is_provisional: bool = Field(default=True, description="True indicates provisional observation, pending reconciliation")
    details: Dict[str, Any] = Field(default_factory=dict)


class ActionExecutor:
    """Executes selected actions safely through idempotent adapters without decision logic."""

    def __init__(
        self,
        idempotency_store: Optional[BaseIdempotencyStore] = None,
        adapters: Optional[Dict[ActionType, BaseActionAdapter]] = None,
        version: str = DEFAULT_EXECUTOR_VERSION,
    ):
        self.store = idempotency_store or InMemoryIdempotencyStore()
        self.version = version
        self.adapters: Dict[ActionType, BaseActionAdapter] = adapters or {
            ActionType.RETRY: PaymentRetryAdapter(),
            ActionType.PAYMENT_UPDATE: PaymentUpdateAdapter(),
            ActionType.REMINDER: ReminderAdapter(),
            ActionType.WAIT: WaitAdapter(),
            ActionType.ESCALATE: EscalateAdapter(),
            ActionType.STOP: StopAdapter(),
        }

    def execute(
        self,
        action: ActionType,
        case: RecoveryCase,
        customer: CustomerProfile,
        decision_id: str,
        attempt: int,
    ) -> ExecutionResult:
        """Execute selected action safely under atomic idempotency synchronization."""
        if action not in self.adapters:
            raise ValueError(f"Unsupported action for execution: {action}")

        # 1. Build exact 4-tuple idempotency key: (case_id, decision_id, action, attempt)
        idemp_key_obj = IdempotencyKey(
            case_id=case.case_id,
            decision_id=decision_id,
            action=action,
            attempt=attempt,
        )
        key_str = idemp_key_obj.to_string()

        # 2. Compute canonical payload hash to detect conflict
        payload_dict = {
            "case_id": case.case_id,
            "customer_id": customer.customer_id,
            "amount_at_risk": float(case.amount_at_risk),
            "action": action.value,
            "decision_id": decision_id,
            "attempt": attempt,
        }
        payload_hash = compute_payload_hash(payload_dict)

        # 3. Atomically execute once under per-key synchronization
        adapter = self.adapters[action]
        record, is_duplicate = self.store.execute_once(
            key=key_str,
            payload_hash=payload_hash,
            execution_fn=lambda: adapter.execute(case, customer),
        )

        status = ExecutionStatus.DUPLICATE if is_duplicate else record.execution_status

        return ExecutionResult(
            case_id=case.case_id,
            decision_id=decision_id,
            action=action,
            attempt=attempt,
            idempotency_key=key_str,
            execution_status=status,
            executor_version=self.version,
            adapter_version=record.adapter_version,
            reference_id=record.reference_id,
            error_code=record.error_code,
            is_duplicate=is_duplicate,
            is_provisional=True,
            details=record.response_data,
        )
