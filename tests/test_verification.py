"""Comprehensive unit tests for Verification and Reconciliation Layer covering all Phase 8 criteria."""
from datetime import datetime, timezone, timedelta
import pytest
from src.domain.actions import ActionType
from src.domain.case import (
    RecoveryCase,
    CustomerProfile,
    PaymentFailureCode,
    PaymentMethodType,
    CaseState,
)
from src.domain.verification import VerificationStatus
from src.execution.status import ExecutionStatus
from src.execution.executor import ExecutionResult
from src.verification.observation import (
    ObservationRecord,
    ObservationConfig,
    ReconciliationStatus,
)
from src.verification.reconciliation import ReconciliationData, ReconciliationConflictError
from src.verification.verifier import VerificationEngine


@pytest.fixture
def base_case() -> RecoveryCase:
    return RecoveryCase(
        case_id="CASE-VERIF-1",
        customer_id="CUST-VERIF-1",
        amount_at_risk=1500.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=3,
        days_waiting=0,
        state=CaseState.ACTIVE,
    )


@pytest.fixture
def mock_success_exec() -> ExecutionResult:
    return ExecutionResult(
        case_id="CASE-VERIF-1",
        decision_id="DEC-VERIF-01",
        action=ActionType.RETRY,
        attempt=1,
        idempotency_key="CASE-VERIF-1:DEC-VERIF-01:RETRY:1",
        execution_status=ExecutionStatus.SUCCESS,
        reference_id="tx_test_123",
        is_duplicate=False,
        is_provisional=True,
    )


# 1. CORE RULE: Execution SUCCESS alone CANNOT produce RESOLVED_RECOVERED
def test_executor_success_does_not_mark_case_recovered(base_case, mock_success_exec):
    """1. CRITICAL TEST: Execution SUCCESS produces provisional observation in IN_OBSERVATION, NOT RESOLVED_RECOVERED."""
    engine = VerificationEngine()
    obs = engine.create_observation(mock_success_exec, base_case)

    # Invariant checks:
    assert obs.is_provisional is True
    assert obs.final_outcome is None
    assert base_case.state == CaseState.IN_OBSERVATION
    assert base_case.state != CaseState.RESOLVED_RECOVERED


# 2. Window enforcement: Before deadline, recovery cannot be finalized
def test_observation_window_enforcement(base_case, mock_success_exec):
    """2. Verify before observation window elapses, reconciliation cannot mark case as recovered."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine(ObservationConfig(default_window_seconds=3600))
    engine.create_observation(mock_success_exec, base_case, start_time=now)

    recon_data = ReconciliationData(
        reconciliation_reference="REC-001",
        settlement_confirmed=True,
        gross_amount_settled=1500.0,
        net_amount_recovered=1500.0,
    )

    # Attempt reconciliation at 10 minutes (before 60 min deadline)
    mid_time = now + timedelta(minutes=10)
    res = engine.reconcile(mock_success_exec.idempotency_key, base_case, recon_data, as_of_time=mid_time)

    assert res.is_window_elapsed(mid_time) is False
    assert res.final_outcome is None
    assert base_case.state != CaseState.RESOLVED_RECOVERED


# 3. SUCCESS + Settlement confirmed after window -> RESOLVED_RECOVERED
def test_successful_settlement_after_window_elapses(base_case, mock_success_exec):
    """3. Verify SUCCESS + window elapsed + settlement confirmed -> RESOLVED_RECOVERED."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine(ObservationConfig(default_window_seconds=3600))
    engine.create_observation(mock_success_exec, base_case, start_time=now)

    recon_data = ReconciliationData(
        reconciliation_reference="REC-SETTLED-001",
        settlement_confirmed=True,
        gross_amount_settled=1500.0,
        net_amount_recovered=1500.0,
    )

    # Reconcile after window (e.g. 70 minutes)
    after_time = now + timedelta(minutes=70)
    res = engine.reconcile(mock_success_exec.idempotency_key, base_case, recon_data, as_of_time=after_time)

    assert res.verification_status == VerificationStatus.SUCCESS
    assert res.reconciliation_status == ReconciliationStatus.SETTLED
    assert res.final_outcome == CaseState.RESOLVED_RECOVERED
    assert base_case.state == CaseState.RESOLVED_RECOVERED
    assert res.is_provisional is False


# 4. Repeated identical reconciliation is idempotent
def test_repeated_identical_reconciliation_is_idempotent(base_case, mock_success_exec):
    """4. Verify repeating identical reconciliation after finalization returns same result without side effects."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine(ObservationConfig(default_window_seconds=3600))
    engine.create_observation(mock_success_exec, base_case, start_time=now)

    recon_data = ReconciliationData(
        reconciliation_reference="REC-SETTLED-001",
        settlement_confirmed=True,
        gross_amount_settled=1500.0,
        net_amount_recovered=1500.0,
    )
    after_time = now + timedelta(minutes=70)

    res1 = engine.reconcile(mock_success_exec.idempotency_key, base_case, recon_data, as_of_time=after_time)
    res2 = engine.reconcile(mock_success_exec.idempotency_key, base_case, recon_data, as_of_time=after_time)

    assert res1.final_outcome == CaseState.RESOLVED_RECOVERED
    assert res2.final_outcome == CaseState.RESOLVED_RECOVERED
    assert res1.reconciliation_fingerprint == res2.reconciliation_fingerprint


# 5. Final recovered + conflicting refund input -> ReconciliationConflictError
def test_final_recovered_conflicting_refund_input_raises_conflict(base_case, mock_success_exec):
    """5. Verify submitting conflicting refund data for already settled observation raises ReconciliationConflictError."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine()
    engine.create_observation(mock_success_exec, base_case, start_time=now, custom_window_seconds=10)

    settle_data = ReconciliationData(
        reconciliation_reference="REC-SETTLED-001",
        settlement_confirmed=True,
        gross_amount_settled=1500.0,
        net_amount_recovered=1500.0,
    )
    after_time = now + timedelta(seconds=20)
    engine.reconcile(mock_success_exec.idempotency_key, base_case, settle_data, as_of_time=after_time)

    # Now attempt to mutate finalized outcome with conflicting refund data
    conflicting_refund = ReconciliationData(
        reconciliation_reference="REC-REFUND-CONFLICT",
        is_refunded=True,
    )

    with pytest.raises(ReconciliationConflictError, match="Reconciliation conflict"):
        engine.reconcile(mock_success_exec.idempotency_key, base_case, conflicting_refund, as_of_time=after_time)

    # State remains immutable
    assert base_case.state == CaseState.RESOLVED_RECOVERED


# 6. Final recovered + conflicting chargeback input -> ReconciliationConflictError
def test_final_recovered_conflicting_chargeback_input_raises_conflict(base_case, mock_success_exec):
    """6. Verify submitting conflicting chargeback data for already settled observation raises ReconciliationConflictError."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine()
    engine.create_observation(mock_success_exec, base_case, start_time=now, custom_window_seconds=10)

    settle_data = ReconciliationData(
        reconciliation_reference="REC-SETTLED-001",
        settlement_confirmed=True,
    )
    after_time = now + timedelta(seconds=20)
    engine.reconcile(mock_success_exec.idempotency_key, base_case, settle_data, as_of_time=after_time)

    conflicting_cb = ReconciliationData(
        reconciliation_reference="REC-CB-CONFLICT",
        is_chargeback=True,
    )

    with pytest.raises(ReconciliationConflictError, match="Reconciliation conflict"):
        engine.reconcile(mock_success_exec.idempotency_key, base_case, conflicting_cb, as_of_time=after_time)


# 7. Final unrecoverable + conflicting settlement input -> ReconciliationConflictError
def test_final_unrecoverable_conflicting_settlement_raises_conflict(base_case, mock_success_exec):
    """7. Verify submitting conflicting settlement data for already refunded/chargebacked observation raises error."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine()
    engine.create_observation(mock_success_exec, base_case, start_time=now, custom_window_seconds=10)

    cb_data = ReconciliationData(
        reconciliation_reference="REC-CB-001",
        is_chargeback=True,
    )
    after_time = now + timedelta(seconds=20)
    engine.reconcile(mock_success_exec.idempotency_key, base_case, cb_data, as_of_time=after_time)

    assert base_case.state == CaseState.RESOLVED_UNRECOVERABLE

    # Attempt to reverse finalized chargeback with settlement
    conflicting_settlement = ReconciliationData(
        reconciliation_reference="REC-REV-001",
        settlement_confirmed=True,
    )

    with pytest.raises(ReconciliationConflictError, match="Reconciliation conflict"):
        engine.reconcile(mock_success_exec.idempotency_key, base_case, conflicting_settlement, as_of_time=after_time)


# 8. Repeated identical refund reconciliation
def test_repeated_identical_refund_reconciliation(base_case, mock_success_exec):
    """8. Verify repeated identical refund reconciliation is idempotent."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine()
    engine.create_observation(mock_success_exec, base_case, start_time=now, custom_window_seconds=10)

    refund_data = ReconciliationData(
        reconciliation_reference="REC-REF-001",
        is_refunded=True,
    )
    after_time = now + timedelta(seconds=20)

    res1 = engine.reconcile(mock_success_exec.idempotency_key, base_case, refund_data, as_of_time=after_time)
    res2 = engine.reconcile(mock_success_exec.idempotency_key, base_case, refund_data, as_of_time=after_time)

    assert res1.final_outcome == CaseState.RESOLVED_UNRECOVERABLE
    assert res2.final_outcome == CaseState.RESOLVED_UNRECOVERABLE


# 9. DECLINE execution outcome handling
def test_decline_execution_handling(base_case):
    """9. Verify DECLINE execution sets verification status to DECLINE and case remains ACTIVE."""
    engine = VerificationEngine()
    decline_exec = ExecutionResult(
        case_id=base_case.case_id,
        decision_id="DEC-DECLINE-01",
        action=ActionType.RETRY,
        attempt=1,
        idempotency_key=f"{base_case.case_id}:DEC-DECLINE-01:RETRY:1",
        execution_status=ExecutionStatus.DECLINE,
        error_code="INSUFFICIENT_FUNDS",
    )
    obs = engine.create_observation(decline_exec, base_case)
    assert obs.verification_status == VerificationStatus.DECLINE
    assert base_case.state == CaseState.ACTIVE
