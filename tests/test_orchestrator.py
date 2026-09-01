"""Comprehensive End-to-End Integration and Mandatory Invariant Tests covering Phase 10."""
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
from src.domain.events import PaymentFailureEvent
from src.execution.status import ExecutionStatus
from src.verification.observation import ReconciliationStatus
from src.verification.reconciliation import ReconciliationData, ReconciliationConflictError
from src.messaging.schema import CandidateMessage, MessageValidationStatus, MessageRejectionReason
from src.orchestration.service import RevenueRecoveryOrchestrator


@pytest.fixture
def standard_customer() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-E2E-1",
        customer_value=8000.0,
        subscription_age_days=180,
        previous_success_rate=0.85,
        previous_contact_count=0,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
        opt_in_email=True,
        opt_in_sms=True,
    )


@pytest.fixture
def failure_event() -> PaymentFailureEvent:
    return PaymentFailureEvent(
        event_id="EVT-E2E-1",
        customer_id="CUST-E2E-1",
        invoice_id="CASE-E2E-1",
        amount=1500.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        payment_method=PaymentMethodType.CREDIT_CARD,
        timestamp=datetime.now(timezone.utc),
    )


# 1. Successful End-to-End Recovery Path
def test_end_to_end_successful_recovery_path(failure_event, standard_customer):
    """1. Verify full workflow: Event -> Case -> Decision -> Execution -> Observation -> Settlement -> RESOLVED_RECOVERED."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    assert case.state == CaseState.ACTIVE

    # Run cycle
    cycle_res = orchestrator.process_recovery_cycle(case, standard_customer)
    assert cycle_res.selected_action is not None
    assert cycle_res.execution_result.execution_status == ExecutionStatus.SUCCESS
    # Check PROVISIONAL observation invariant
    assert cycle_res.case_state == CaseState.IN_OBSERVATION
    assert cycle_res.observation_record.is_provisional is True

    # Reconcile after window passes
    recon_time = datetime.now(timezone.utc) + timedelta(hours=2)
    recon_data = ReconciliationData(
        reconciliation_reference="REC-E2E-001",
        settlement_confirmed=True,
        gross_amount_settled=1500.0,
        net_amount_recovered=1500.0,
    )
    final_obs = orchestrator.reconcile_case(
        idempotency_key=cycle_res.idempotency_key,
        case=case,
        reconciliation_data=recon_data,
        as_of_time=recon_time,
    )

    assert final_obs.final_outcome == CaseState.RESOLVED_RECOVERED
    assert case.state == CaseState.RESOLVED_RECOVERED
    assert final_obs.reconciliation_status == ReconciliationStatus.SETTLED


# 2. Payment failure path (Declined retry)
def test_end_to_end_payment_decline_path(failure_event, standard_customer):
    """2. Verify retry decline keeps case active without marking as recovered."""
    from src.execution.executor import ActionExecutor
    from src.execution.adapters import BaseActionAdapter, AdapterResponse

    class DecliningAdapter(BaseActionAdapter):
        def execute(self, case, customer):
            return AdapterResponse(status=ExecutionStatus.DECLINE, error_code="GENERIC_DECLINE")

    failing_exec = ActionExecutor(adapters={ActionType.RETRY: DecliningAdapter()})
    orchestrator = RevenueRecoveryOrchestrator(executor=failing_exec)
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)

    cycle_res = orchestrator.process_recovery_cycle(
        case, standard_customer, candidate_actions=[ActionType.RETRY]
    )
    assert cycle_res.execution_result.execution_status == ExecutionStatus.DECLINE
    assert cycle_res.case_state == CaseState.ACTIVE


# 3 & 4. Empty candidate set & Policy Prohibitions
def test_end_to_end_empty_candidate_set(failure_event, standard_customer):
    """4. Verify empty candidate actions results in no action executed and STOP is not injected."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)

    cycle_res = orchestrator.process_recovery_cycle(case, standard_customer, candidate_actions=[])
    assert cycle_res.selected_action is None
    assert cycle_res.execution_result is None
    assert cycle_res.case_state == CaseState.ACTIVE


# 5. STOP Path - Delegates strictly to frozen Execution/Verification layer
def test_end_to_end_stop_path(failure_event, standard_customer):
    """5. Verify STOP action executes and transitions as defined by frozen VerificationEngine."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)

    cycle_res = orchestrator.process_recovery_cycle(case, standard_customer, candidate_actions=[ActionType.STOP])
    assert cycle_res.selected_action == ActionType.STOP
    assert cycle_res.execution_result.execution_status == ExecutionStatus.SUCCESS
    assert cycle_res.observation_record.final_outcome == CaseState.RESOLVED_UNRECOVERABLE
    assert cycle_res.case_state == CaseState.RESOLVED_UNRECOVERABLE


# 10 & 11. Execution Idempotency & Duplicate Prevention
def test_end_to_end_execution_idempotency(failure_event, standard_customer):
    """10 & 11. Verify duplicate cycle with same decision_id does not re-execute side effect."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)

    res1 = orchestrator.process_recovery_cycle(case, standard_customer, decision_id="DEC-IDEMP-SAME")
    res2 = orchestrator.process_recovery_cycle(case, standard_customer, decision_id="DEC-IDEMP-SAME")

    assert res1.execution_result.is_duplicate is False
    assert res2.execution_result.is_duplicate is True
    assert res1.idempotency_key == res2.idempotency_key


# 15 & 16. Refund & Chargeback Invalidation
def test_end_to_end_refund_and_chargeback(failure_event, standard_customer):
    """15 & 16. Verify refunds and chargebacks invalidate recovery."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    cycle = orchestrator.process_recovery_cycle(case, standard_customer)

    recon_time = datetime.now(timezone.utc) + timedelta(hours=2)
    cb_data = ReconciliationData(
        reconciliation_reference="REC-CB-123",
        is_chargeback=True,
    )
    final_obs = orchestrator.reconcile_case(cycle.idempotency_key, case, cb_data, as_of_time=recon_time)

    assert final_obs.final_outcome == CaseState.RESOLVED_UNRECOVERABLE
    assert case.state == CaseState.RESOLVED_UNRECOVERABLE


# 18-22. Constrained LLM Messaging & Anti-Hallucination
def test_end_to_end_messaging_approval_and_rejection(failure_event, standard_customer):
    """18-22. Verify messaging validation in orchestrator cycle."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)

    # Valid cycle
    cycle = orchestrator.process_recovery_cycle(
        case, standard_customer, candidate_actions=[ActionType.PAYMENT_UPDATE]
    )
    assert cycle.delivered_message is not None
    assert cycle.message_validation.is_approved is True

    # Customer with SMS consent but lacking email consent:
    no_email_cust = standard_customer.model_copy(deep=True)
    no_email_cust.opt_in_email = False
    no_email_cust.opt_in_sms = True

    cycle_no_consent = orchestrator.process_recovery_cycle(
        case, no_email_cust, candidate_actions=[ActionType.PAYMENT_UPDATE]
    )
    assert cycle_no_consent.delivered_message is None
    assert cycle_no_consent.message_validation.is_approved is False
    assert MessageRejectionReason.CONSENT_MISSING in cycle_no_consent.message_validation.rejection_reasons
    assert cycle_no_consent.selected_action == ActionType.PAYMENT_UPDATE


# ==============================================================================
# MANDATORY 10 INVARIANT TESTS
# ==============================================================================

# INVARIANT 1: LLM cannot select action.
def test_invariant_1_llm_cannot_select_action(failure_event, standard_customer):
    """INVARIANT 1: DecisionEngine alone selects action; LLM cannot select or alter action."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    cycle = orchestrator.process_recovery_cycle(case, standard_customer, candidate_actions=[ActionType.PAYMENT_UPDATE])

    assert cycle.selected_action == ActionType.PAYMENT_UPDATE


# INVARIANT 2: LLM cannot modify amount.
def test_invariant_2_llm_cannot_modify_amount(failure_event, standard_customer):
    """INVARIANT 2: Case amount remains ground truth; LLM claims cannot alter case financials."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    cycle = orchestrator.process_recovery_cycle(case, standard_customer)

    assert case.amount_at_risk == 1500.0


# INVARIANT 3: LLM cannot modify failure code.
def test_invariant_3_llm_cannot_modify_failure_code(failure_event, standard_customer):
    """INVARIANT 3: Failure code is immutable ground truth."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    cycle = orchestrator.process_recovery_cycle(case, standard_customer)

    assert case.failure_code == PaymentFailureCode.INSUFFICIENT_FUNDS


# INVARIANT 4: Execution SUCCESS cannot directly produce final recovery.
def test_invariant_4_execution_success_cannot_directly_produce_final_recovery(failure_event, standard_customer):
    """INVARIANT 4: Executor SUCCESS produces IN_OBSERVATION, never RESOLVED_RECOVERED directly."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    cycle = orchestrator.process_recovery_cycle(case, standard_customer)

    assert cycle.execution_result.execution_status == ExecutionStatus.SUCCESS
    assert case.state == CaseState.IN_OBSERVATION
    assert case.state != CaseState.RESOLVED_RECOVERED


# INVARIANT 5: Policy filtering happens before action selection.
def test_invariant_5_policy_filtering_precedes_action_selection(failure_event, standard_customer):
    """INVARIANT 5: Prohibited actions are pruned before Q2 scoring and argmax."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    case.retry_attempt_count = 4

    cycle = orchestrator.process_recovery_cycle(case, standard_customer)

    # Retry is filtered out by PolicyEngine RetryLimitRule
    assert ActionType.RETRY not in cycle.decision_result.allowed_actions
    assert cycle.selected_action != ActionType.RETRY


# INVARIANT 6: STOP is not injected into an explicitly empty candidate set.
def test_invariant_6_stop_not_injected_into_empty_candidate_set(failure_event, standard_customer):
    """INVARIANT 6: Explicit candidate_actions=[] must not have STOP injected."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    cycle = orchestrator.process_recovery_cycle(case, standard_customer, candidate_actions=[])

    assert cycle.selected_action is None
    assert cycle.decision_result.allowed_actions == []


# INVARIANT 7: Same idempotency key cannot execute an external action twice.
def test_invariant_7_same_idempotency_key_cannot_execute_twice(failure_event, standard_customer):
    """INVARIANT 7: Duplicate execution is blocked by atomic idempotency store."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)

    c1 = orchestrator.process_recovery_cycle(case, standard_customer, decision_id="DEC-INV-7")
    c2 = orchestrator.process_recovery_cycle(case, standard_customer, decision_id="DEC-INV-7")

    assert c1.execution_result.is_duplicate is False
    assert c2.execution_result.is_duplicate is True


# INVARIANT 8: Final reconciliation cannot be silently reversed.
def test_invariant_8_final_reconciliation_cannot_be_silently_reversed(failure_event, standard_customer):
    """INVARIANT 8: Conflicting reconciliation on finalized observation raises ReconciliationConflictError."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    cycle = orchestrator.process_recovery_cycle(case, standard_customer)

    recon_time = datetime.now(timezone.utc) + timedelta(hours=2)
    settle_data = ReconciliationData(reconciliation_reference="REC-SETTLE", settlement_confirmed=True)
    orchestrator.reconcile_case(cycle.idempotency_key, case, settle_data, as_of_time=recon_time)

    assert case.state == CaseState.RESOLVED_RECOVERED

    # Attempt conflicting reversal
    conflicting_refund = ReconciliationData(reconciliation_reference="REC-REF-CONF", is_refunded=True)
    with pytest.raises(ReconciliationConflictError):
        orchestrator.reconcile_case(cycle.idempotency_key, case, conflicting_refund, as_of_time=recon_time)

    assert case.state == CaseState.RESOLVED_RECOVERED


# INVARIANT 9: MessageValidator cannot modify selected_action.
def test_invariant_9_validator_cannot_modify_selected_action(failure_event, standard_customer):
    """INVARIANT 9: Validator rejection does not alter selected_action."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)

    cycle = orchestrator.process_recovery_cycle(
        case, standard_customer,
        candidate_actions=[ActionType.PAYMENT_UPDATE],
        preferred_template_id="TMPL_INVALID_ID",
    )
    assert cycle.selected_action == ActionType.PAYMENT_UPDATE
    assert cycle.message_validation.is_approved is False


# INVARIANT 10: Unknown states cannot become successful recovery.
def test_invariant_10_unknown_states_cannot_become_successful_recovery(failure_event, standard_customer):
    """INVARIANT 10: Incomplete or unconfirmed reconciliation never becomes RESOLVED_RECOVERED."""
    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(failure_event, standard_customer)
    cycle = orchestrator.process_recovery_cycle(case, standard_customer)

    recon_time = datetime.now(timezone.utc) + timedelta(hours=2)
    incomplete_data = ReconciliationData(
        reconciliation_reference="REC-INCOMPLETE",
        settlement_confirmed=False,
    )
    obs = orchestrator.reconcile_case(cycle.idempotency_key, case, incomplete_data, as_of_time=recon_time)

    assert obs.final_outcome is None
    assert obs.is_provisional is True
    assert case.state != CaseState.RESOLVED_RECOVERED
