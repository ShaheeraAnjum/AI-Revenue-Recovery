"""Comprehensive unit tests for Policy & Safety Engine."""
import pytest
from src.domain.actions import ActionType
from src.domain.case import (
    RecoveryCase,
    CustomerProfile,
    PaymentFailureCode,
    PaymentMethodType,
    CaseState,
)
from src.policy.config import PolicyConfig
from src.policy.rules import PolicyRejectionReason
from src.policy.engine import PolicyEngine


@pytest.fixture
def standard_customer() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-POL-1",
        customer_value=5000.0,
        subscription_age_days=180,
        previous_success_rate=0.8,
        previous_contact_count=0,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
        opt_in_email=True,
        opt_in_sms=True,
        active_recovery_cases=1,
    )


@pytest.fixture
def standard_case() -> RecoveryCase:
    return RecoveryCase(
        case_id="CASE-POL-1",
        customer_id="CUST-POL-1",
        amount_at_risk=1500.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=5,
        state=CaseState.ACTIVE,
        retry_attempt_count=0,
        reminder_count=0,
        days_waiting=0,
    )


def test_retry_limit_reached(standard_case, standard_customer):
    """1. Test retry limit reached: RETRY is prohibited with exact reason."""
    engine = PolicyEngine(PolicyConfig(max_retries_per_case=3))
    standard_case.retry_attempt_count = 3

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY not in decision.allowed_actions
    assert ActionType.RETRY in decision.prohibited_actions
    assert "retry_limit_exceeded" in decision.prohibited_actions[ActionType.RETRY]


def test_retry_still_allowed(standard_case, standard_customer):
    """2. Test retry still allowed when attempt count is below limit."""
    engine = PolicyEngine(PolicyConfig(max_retries_per_case=3))
    standard_case.retry_attempt_count = 2

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY in decision.allowed_actions
    assert ActionType.RETRY not in decision.prohibited_actions


def test_communication_consent_absent(standard_case, standard_customer):
    """3. Test customer consent missing: REMINDER and PAYMENT_UPDATE are prohibited."""
    engine = PolicyEngine()
    standard_customer.opt_in_email = False
    standard_customer.opt_in_sms = False

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER not in decision.allowed_actions
    assert ActionType.PAYMENT_UPDATE not in decision.allowed_actions
    assert "consent_missing" in decision.prohibited_actions[ActionType.REMINDER]
    assert "consent_missing" in decision.prohibited_actions[ActionType.PAYMENT_UPDATE]


def test_communication_consent_present(standard_case, standard_customer):
    """4. Test communication consent present: REMINDER allowed."""
    engine = PolicyEngine()
    standard_customer.opt_in_email = True
    standard_customer.opt_in_sms = False

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER in decision.allowed_actions
    assert ActionType.PAYMENT_UPDATE in decision.allowed_actions


def test_contact_limit_reached(standard_case, standard_customer):
    """5. Test contact limit reached: REMINDER and PAYMENT_UPDATE prohibited."""
    engine = PolicyEngine(PolicyConfig(max_total_contacts_per_customer=5))
    standard_customer.previous_contact_count = 5

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER not in decision.allowed_actions
    assert ActionType.PAYMENT_UPDATE not in decision.allowed_actions
    assert "contact_limit" in decision.prohibited_actions[ActionType.REMINDER]


def test_contact_still_allowed(standard_case, standard_customer):
    """6. Test contact still allowed below limit."""
    engine = PolicyEngine(PolicyConfig(max_total_contacts_per_customer=5))
    standard_customer.previous_contact_count = 4

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER in decision.allowed_actions


def test_payment_network_rule_hard_decline(standard_case, standard_customer):
    """7. Test network rule prohibiting retry on hard declines like FRAUD_SUSPECTED."""
    engine = PolicyEngine()
    standard_case.failure_code = PaymentFailureCode.FRAUD_SUSPECTED

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY not in decision.allowed_actions
    assert "network_rule_hard_decline" in decision.prohibited_actions[ActionType.RETRY]

    # CARD_EXPIRED is also a hard decline
    standard_case.failure_code = PaymentFailureCode.CARD_EXPIRED
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY not in decision2.allowed_actions
    assert "network_rule_hard_decline" in decision2.prohibited_actions[ActionType.RETRY]


def test_wait_limit_reached(standard_case, standard_customer):
    """8. Test WAIT limit reached after max consecutive days."""
    engine = PolicyEngine(PolicyConfig(max_consecutive_wait_days=14))
    standard_case.days_waiting = 14

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.WAIT not in decision.allowed_actions
    assert "wait_limit_exceeded" in decision.prohibited_actions[ActionType.WAIT]


def test_escalate_restriction_thresholds(standard_case, standard_customer):
    """9. Test ESCALATE constraints on aging and minimum amount."""
    engine = PolicyEngine(PolicyConfig(
        min_days_overdue_for_escalation=3,
        min_amount_for_escalation=500.0,
    ))

    # Case too fresh (days_overdue < 3)
    standard_case.days_overdue = 1
    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.ESCALATE not in decision.allowed_actions
    assert "escalation_aging_insufficient" in decision.prohibited_actions[ActionType.ESCALATE]

    # Case amount too small (amount_at_risk < 500)
    standard_case.days_overdue = 10
    standard_case.amount_at_risk = 50.0
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.ESCALATE not in decision2.allowed_actions
    assert "escalation_amount_insufficient" in decision2.prohibited_actions[ActionType.ESCALATE]

    # Meets all criteria
    standard_case.days_overdue = 10
    standard_case.amount_at_risk = 2000.0
    decision3 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.ESCALATE in decision3.allowed_actions


def test_stop_action_eligibility(standard_case, standard_customer):
    """10. Test STOP action is always eligible during active recovery."""
    engine = PolicyEngine()
    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.STOP in decision.allowed_actions


def test_multiple_actions_prohibited_simultaneously(standard_case, standard_customer):
    """11. Test multiple simultaneous prohibitions."""
    engine = PolicyEngine(PolicyConfig(
        max_retries_per_case=2,
        max_consecutive_wait_days=7,
        min_days_overdue_for_escalation=10,
    ))
    standard_case.retry_attempt_count = 2
    standard_case.days_waiting = 7
    standard_case.days_overdue = 2
    standard_customer.opt_in_email = False
    standard_customer.opt_in_sms = False

    decision = engine.evaluate(standard_case, standard_customer)

    # RETRY, WAIT, ESCALATE, REMINDER, PAYMENT_UPDATE should all be prohibited
    assert ActionType.RETRY in decision.prohibited_actions
    assert ActionType.WAIT in decision.prohibited_actions
    assert ActionType.ESCALATE in decision.prohibited_actions
    assert ActionType.REMINDER in decision.prohibited_actions
    assert ActionType.PAYMENT_UPDATE in decision.prohibited_actions

    # ONLY STOP remains allowed
    assert decision.allowed_actions == [ActionType.STOP]


def test_determinism_identical_inputs(standard_case, standard_customer):
    """12. Test identical inputs produce bit-identical allowed and prohibited actions."""
    engine = PolicyEngine()
    dec1 = engine.evaluate(standard_case, standard_customer)
    dec2 = engine.evaluate(standard_case, standard_customer)

    assert dec1.allowed_actions == dec2.allowed_actions
    assert dec1.prohibited_actions == dec2.prohibited_actions
    assert dec1.policy_version == dec2.policy_version


def test_policy_version_propagation(standard_case, standard_customer):
    """13. Test policy version string propagation."""
    engine = PolicyEngine(PolicyConfig(policy_version="policy_v5.2.1-custom"))
    decision = engine.evaluate(standard_case, standard_customer)
    assert decision.policy_version == "policy_v5.2.1-custom"
