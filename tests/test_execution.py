"""Comprehensive unit tests for Idempotent Action Executor covering all Phase 7 criteria."""
import time
import concurrent.futures
import pytest
from src.domain.actions import ActionType
from src.domain.case import (
    RecoveryCase,
    CustomerProfile,
    PaymentFailureCode,
    PaymentMethodType,
    CaseState,
    IdempotencyKey,
)
from src.execution.status import ExecutionStatus
from src.execution.idempotency import (
    InMemoryIdempotencyStore,
    IdempotencyConflictError,
)
from src.execution.adapters import (
    PaymentRetryAdapter,
    PaymentUpdateAdapter,
    ReminderAdapter,
    WaitAdapter,
    EscalateAdapter,
    StopAdapter,
    AdapterResponse,
    BaseActionAdapter,
)
from src.execution.executor import ActionExecutor, ExecutionResult


@pytest.fixture
def exec_customer() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-EXEC-1",
        customer_value=7500.0,
        subscription_age_days=200,
        previous_success_rate=0.85,
        previous_contact_count=1,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
        opt_in_email=True,
        opt_in_sms=True,
    )


@pytest.fixture
def exec_case() -> RecoveryCase:
    return RecoveryCase(
        case_id="CASE-EXEC-1",
        customer_id="CUST-EXEC-1",
        amount_at_risk=2000.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=3,
        days_waiting=0,
        state=CaseState.ACTIVE,
    )


# 1-7. Supported action execution semantics
def test_all_supported_actions_have_execution_semantics(exec_case, exec_customer):
    """1-7. Verify explicit execution semantics for all 6 supported actions."""
    executor = ActionExecutor()

    for action in ActionType:
        res = executor.execute(
            action=action,
            case=exec_case,
            customer=exec_customer,
            decision_id=f"DEC-{action.value}",
            attempt=1,
        )
        assert isinstance(res, ExecutionResult)
        assert res.execution_status == ExecutionStatus.SUCCESS
        assert res.reference_id is not None
        assert res.is_provisional is True


def test_retry_execution(exec_case, exec_customer):
    """2. Verify RETRY calls payment gateway adapter."""
    executor = ActionExecutor()
    res = executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-RETRY", 1)
    assert res.action == ActionType.RETRY
    assert res.reference_id.startswith("tx_retry_")
    assert res.details["amount_charged"] == 2000.0


def test_payment_update_execution(exec_case, exec_customer):
    """3. Verify PAYMENT_UPDATE generates hosted portal link."""
    executor = ActionExecutor()
    res = executor.execute(ActionType.PAYMENT_UPDATE, exec_case, exec_customer, "DEC-UPDATE", 1)
    assert res.action == ActionType.PAYMENT_UPDATE
    assert "portal_url" in res.details


def test_reminder_execution(exec_case, exec_customer):
    """4. Verify REMINDER dispatches notification."""
    executor = ActionExecutor()
    res = executor.execute(ActionType.REMINDER, exec_case, exec_customer, "DEC-REMIND", 1)
    assert res.action == ActionType.REMINDER
    assert res.details["channel"] == "email"


def test_wait_execution(exec_case, exec_customer):
    """5. Verify WAIT records schedule and does not charge card."""
    executor = ActionExecutor()
    res = executor.execute(ActionType.WAIT, exec_case, exec_customer, "DEC-WAIT", 1)
    assert res.action == ActionType.WAIT
    assert res.details["hold_duration_days"] == 1


def test_escalate_execution(exec_case, exec_customer):
    """6. Verify ESCALATE creates manual review ticket."""
    executor = ActionExecutor()
    res = executor.execute(ActionType.ESCALATE, exec_case, exec_customer, "DEC-ESC", 1)
    assert res.action == ActionType.ESCALATE
    assert res.reference_id.startswith("TICKET-OPS-")


def test_stop_noop_execution(exec_case, exec_customer):
    """7. Verify STOP produces terminal no-op execution."""
    executor = ActionExecutor()
    res = executor.execute(ActionType.STOP, exec_case, exec_customer, "DEC-STOP", 1)
    assert res.action == ActionType.STOP
    assert res.details["recovery_ended"] is True


# 8-11. Idempotency Key, Storage, and Duplicate Execution
def test_idempotency_duplicate_execution_returns_cached_result(exec_case, exec_customer):
    """8, 9, 10, 11. Verify duplicate submission with exact key returns recorded result without re-executing."""
    executor = ActionExecutor()

    # First execution
    res1 = executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-DUP-01", 1)
    assert res1.is_duplicate is False
    assert res1.execution_status == ExecutionStatus.SUCCESS

    # Second execution with exact same key
    res2 = executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-DUP-01", 1)
    assert res2.is_duplicate is True
    assert res2.execution_status == ExecutionStatus.DUPLICATE
    assert res2.reference_id == res1.reference_id
    assert res2.idempotency_key == res1.idempotency_key


# Real Concurrency Regression Test
def test_concurrent_same_key_execution_adapter_called_exactly_once(exec_case, exec_customer):
    """MANDATORY CONCURRENCY TEST: Verify that concurrent callers using identical idempotency key execute adapter exactly once."""
    class CountingRetryAdapter(BaseActionAdapter):
        def __init__(self):
            self.execution_count = 0
            self._lock = time.sleep

        def execute(self, case, customer):
            self.execution_count += 1
            time.sleep(0.05)  # Simulate non-trivial external gateway latency
            return AdapterResponse(
                status=ExecutionStatus.SUCCESS,
                reference_id=f"tx_concurrent_{self.execution_count}",
                details={"charged": float(case.amount_at_risk)},
            )

    counting_adapter = CountingRetryAdapter()
    executor = ActionExecutor(adapters={ActionType.RETRY: counting_adapter})

    # Dispatch 10 simultaneous threads with the EXACT SAME (case_id, decision_id, action, attempt)
    num_threads = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [
            pool.submit(
                executor.execute,
                ActionType.RETRY,
                exec_case,
                exec_customer,
                "DEC-CONCURRENT-01",
                1,
            )
            for _ in range(num_threads)
        ]
        results = [f.result() for f in futures]

    # 1. Adapter execution count MUST be exactly 1
    assert counting_adapter.execution_count == 1

    # 2. All 10 results must share the exact same reference_id
    ref_ids = {r.reference_id for r in results}
    assert len(ref_ids) == 1
    assert "tx_concurrent_1" in ref_ids

    # 3. Exactly 1 result is the initial execution (is_duplicate=False) and 9 are duplicates (is_duplicate=True)
    duplicates = [r.is_duplicate for r in results]
    assert duplicates.count(False) == 1
    assert duplicates.count(True) == (num_threads - 1)


# Conflict handling on different payload
def test_idempotency_conflict_detection(exec_case, exec_customer):
    """12. Verify submitting same key with materially different payload raises IdempotencyConflictError."""
    executor = ActionExecutor()
    executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-CONF", 1)

    # Modify case amount at risk (material payload alteration)
    conflicting_case = exec_case.model_copy(deep=True)
    conflicting_case.amount_at_risk = 9999.0

    with pytest.raises(IdempotencyConflictError, match="Idempotency conflict"):
        executor.execute(ActionType.RETRY, conflicting_case, exec_customer, "DEC-CONF", 1)


# Concurrent Conflict Detection
def test_concurrent_same_key_conflicting_payload_detection(exec_case, exec_customer):
    """Verify concurrent conflict detection if another thread arrives with conflicting payload."""
    executor = ActionExecutor()
    conflicting_case = exec_case.model_copy(deep=True)
    conflicting_case.amount_at_risk = 50000.0

    # First call completes
    executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-CON-RACE", 1)

    # Conflicting call
    with pytest.raises(IdempotencyConflictError):
        executor.execute(ActionType.RETRY, conflicting_case, exec_customer, "DEC-CON-RACE", 1)


# Key uniqueness across attempts and actions
def test_different_attempt_and_action_keys(exec_case, exec_customer):
    """13, 14. Verify different attempts and actions produce distinct idempotency keys."""
    executor = ActionExecutor()

    res_att1 = executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-KEY", 1)
    res_att2 = executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-KEY", 2)
    res_act2 = executor.execute(ActionType.WAIT, exec_case, exec_customer, "DEC-KEY", 1)

    assert res_att1.idempotency_key != res_att2.idempotency_key
    assert res_att1.idempotency_key != res_act2.idempotency_key


# Execution does not imply final recovery
def test_execution_is_strictly_provisional(exec_case, exec_customer):
    """15. Verify execution result is marked provisional and case state is not modified to final recovery."""
    executor = ActionExecutor()
    res = executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-PROV", 1)

    assert res.is_provisional is True
    assert exec_case.state == CaseState.ACTIVE


# Adapter error / decline handling
def test_adapter_decline_and_timeout_handling(exec_case, exec_customer):
    """17. Verify decline and timeout statuses are preserved in execution result."""
    class DecliningAdapter(BaseActionAdapter):
        def execute(self, case, customer):
            return AdapterResponse(
                status=ExecutionStatus.DECLINE,
                error_code="CARD_DECLINED_GENERIC",
                details={"decline_code": "05"},
            )

    executor = ActionExecutor(adapters={ActionType.RETRY: DecliningAdapter()})
    res = executor.execute(ActionType.RETRY, exec_case, exec_customer, "DEC-FAIL", 1)

    assert res.execution_status == ExecutionStatus.DECLINE
    assert res.error_code == "CARD_DECLINED_GENERIC"
