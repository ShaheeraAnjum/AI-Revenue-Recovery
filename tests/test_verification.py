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
from src.verification.reconciliation import ReconciliationData
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
    obs = engine.create_observation(mock_success_exec, base_case, start_time=now)

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
    obs = engine.create_observation(mock_success_exec, base_case, start_time=now)

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


# 4. Refund handling invalidates recovery
def test_refund_invalidates_recovery(base_case, mock_success_exec):
    """4. Verify SUCCESS + refund -> RECONCILIATION status and RESOLVED_UNRECOVERABLE."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine()
    obs = engine.create_observation(mock_success_exec, base_case, start_time=now, custom_window_seconds=100)

    recon_data = ReconciliationData(
        reconciliation_reference="REC-REFUND-001",
        settlement_confirmed=True,
        is_refunded=True,
        gross_amount_settled=1500.0,
        net_amount_recovered=0.0,
    )

    after_time = now + timedelta(seconds=200)
    res = engine.reconcile(mock_success_exec.idempotency_key, base_case, recon_data, as_of_time=after_time)

    assert res.reconciliation_status == ReconciliationStatus.REFUNDED
    assert res.final_outcome == CaseState.RESOLVED_UNRECOVERABLE
    assert base_case.state == CaseState.RESOLVED_UNRECOVERABLE


# 5. Chargeback handling invalidates recovery
def test_chargeback_invalidates_recovery(base_case, mock_success_exec):
    """5. Verify SUCCESS + chargeback -> CHARGEBACK status and RESOLVED_UNRECOVERABLE."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine()
    engine.create_observation(mock_success_exec, base_case, start_time=now, custom_window_seconds=100)

    recon_data = ReconciliationData(
        reconciliation_reference="REC-CB-001",
        settlement_confirmed=False,
        is_chargeback=True,
    )

    after_time = now + timedelta(seconds=200)
    res = engine.reconcile(mock_success_exec.idempotency_key, base_case, recon_data, as_of_time=after_time)

    assert res.reconciliation_status == ReconciliationStatus.CHARGEBACK
    assert res.final_outcome == CaseState.RESOLVED_UNRECOVERABLE
    assert base_case.state == CaseState.RESOLVED_UNRECOVERABLE


# 6. DECLINE execution outcome handling
def test_decline_execution_handling(base_case):
    """6. Verify DECLINE execution sets verification status to DECLINE and case remains ACTIVE."""
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


# 7. Repeated reconciliation is deterministic without duplicate side effects
def test_repeated_reconciliation_determinism(base_case, mock_success_exec):
    """7. Verify repeated reconciliation of the same key returns consistent, bit-exact result."""
    now = datetime.now(timezone.utc)
    engine = VerificationEngine()
    engine.create_observation(mock_success_exec, base_case, start_time=now, custom_window_seconds=50)

    recon_data = ReconciliationData(
        reconciliation_reference="REC-DET-01",
        settlement_confirmed=True,
        net_amount_recovered=1500.0,
    )
    after_time = now + timedelta(seconds=100)

    res1 = engine.reconcile(mock_success_exec.idempotency_key, base_case, recon_data, as_of_time=after_time)
    res2 = engine.reconcile(mock_success_exec.idempotency_key, base_case, recon_data, as_of_time=after_time)

    assert res1.final_outcome == res2.final_outcome
    assert res1.verification_status == res2.verification_status
    assert res1.reconciliation_status == res2.reconciliation_status
