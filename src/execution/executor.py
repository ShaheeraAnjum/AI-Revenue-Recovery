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
        """Execute selected action safely under exact idempotency key."""
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

        # 3. Check idempotency store for existing execution
        existing = self.store.get(key_str)
        if existing is not None:
            if existing.payload_hash != payload_hash:
                raise IdempotencyConflictError(
                    f"Idempotency conflict: key '{key_str}' previously submitted with different payload"
                )
            # Return previously recorded execution result marked as DUPLICATE
            return ExecutionResult(
                case_id=case.case_id,
                decision_id=decision_id,
                action=action,
                attempt=attempt,
                idempotency_key=key_str,
                execution_status=ExecutionStatus.DUPLICATE,
                executor_version=self.version,
                adapter_version=DEFAULT_ADAPTER_VERSION,
                reference_id=existing.reference_id,
                error_code=existing.error_code,
                is_duplicate=True,
                is_provisional=True,
                details=existing.response_data,
            )

        # 4. Dispatch to appropriate adapter
        adapter = self.adapters[action]
        adapter_resp = adapter.execute(case, customer)

        # 5. Record result in idempotency store
        record = IdempotencyRecord(
            key=key_str,
            payload_hash=payload_hash,
            execution_status=adapter_resp.status,
            response_data=adapter_resp.details,
            reference_id=adapter_resp.reference_id,
            error_code=adapter_resp.error_code,
        )
        self.store.put(record)

        # 6. Return fresh execution result (provisional observation)
        return ExecutionResult(
            case_id=case.case_id,
            decision_id=decision_id,
            action=action,
            attempt=attempt,
            idempotency_key=key_str,
            execution_status=adapter_resp.status,
            executor_version=self.version,
            adapter_version=adapter_resp.adapter_version,
            reference_id=adapter_resp.reference_id,
            error_code=adapter_resp.error_code,
            is_duplicate=False,
            is_provisional=True,
            details=adapter_resp.details,
        )
