"""Comprehensive unit tests for Policy & Safety Engine covering 100% of safety parameters."""
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
        escalation_count=0,
        days_waiting=0,
        is_pci_tokenized=True,
    )


# 1. Retry limits
def test_retry_limit_reached(standard_case, standard_customer):
    """1. Test retry limit reached: RETRY prohibited with exact reason."""
    engine = PolicyEngine(PolicyConfig(max_retries_per_case=3))
    standard_case.retry_attempt_count = 3

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY not in decision.allowed_actions
    assert ActionType.RETRY in decision.prohibited_actions
    assert "retry_limit_exceeded" in decision.prohibited_actions[ActionType.RETRY]


def test_retry_still_allowed(standard_case, standard_customer):
    """2. Test retry still allowed below limit."""
    engine = PolicyEngine(PolicyConfig(max_retries_per_case=3))
    standard_case.retry_attempt_count = 2

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY in decision.allowed_actions
    assert ActionType.RETRY not in decision.prohibited_actions


# 2. Card network rules
def test_payment_network_rule_hard_decline(standard_case, standard_customer):
    """3. Test network rule prohibiting retry on hard declines like FRAUD_SUSPECTED."""
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


# 3. PCI Boundary enforcement
def test_pci_tokenization_boundary_enforcement(standard_case, standard_customer):
    """4. Test PCI tokenization boundary: non-tokenized payment credentials block automated RETRY."""
    engine = PolicyEngine(PolicyConfig(enforce_pci_tokenization_boundary=True))
    standard_case.is_pci_tokenized = False

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY not in decision.allowed_actions
    assert "pci_tokenization_boundary_violation" in decision.prohibited_actions[ActionType.RETRY]

    # Tokenized payment allows RETRY
    standard_case.is_pci_tokenized = True
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY in decision2.allowed_actions


# 4. VIP Exploration limits
def test_vip_exploration_protection(standard_case, standard_customer):
    """5. Test VIP protection against repeated speculative retries on high-value accounts."""
    engine = PolicyEngine(PolicyConfig(
        enable_exploration_protection_for_vip=True,
        vip_customer_value_threshold=25000.0,
    ))
    standard_customer.customer_value = 30000.0  # VIP customer
    standard_case.retry_attempt_count = 1       # Prior attempt failed

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY not in decision.allowed_actions
    assert "vip_exploration_protection_active" in decision.prohibited_actions[ActionType.RETRY]

    # Below threshold -> not restricted
    standard_customer.customer_value = 10000.0
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY in decision2.allowed_actions

    # VIP protection disabled -> normal behavior
    engine_disabled = PolicyEngine(PolicyConfig(
        enable_exploration_protection_for_vip=False,
        vip_customer_value_threshold=25000.0,
    ))
    standard_customer.customer_value = 50000.0
    decision3 = engine_disabled.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY in decision3.allowed_actions


# 5. Escalation limit & threshold enforcement
def test_escalation_limit_enforcement(standard_case, standard_customer):
    """6. Test ESCALATE limit enforcement on escalation count."""
    engine = PolicyEngine(PolicyConfig(
        max_escalations_per_case=1,
        min_days_overdue_for_escalation=3,
        min_amount_for_escalation=500.0,
    ))
    standard_case.days_overdue = 5
    standard_case.amount_at_risk = 2000.0
    
    # 0 previous escalations -> eligible
    standard_case.escalation_count = 0
    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.ESCALATE in decision.allowed_actions

    # 1 previous escalation (limit reached) -> prohibited
    standard_case.escalation_count = 1
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.ESCALATE not in decision2.allowed_actions
    assert "escalation_limit_exceeded" in decision2.prohibited_actions[ActionType.ESCALATE]

    # >1 previous escalations (above limit) -> prohibited
    standard_case.escalation_count = 2
    decision3 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.ESCALATE not in decision3.allowed_actions
    assert "escalation_limit_exceeded" in decision3.prohibited_actions[ActionType.ESCALATE]


def test_escalation_aging_and_amount_thresholds(standard_case, standard_customer):
    """7. Test ESCALATE aging and amount thresholds."""
    engine = PolicyEngine(PolicyConfig(
        min_days_overdue_for_escalation=3,
        min_amount_for_escalation=500.0,
    ))

    # Case too fresh (days_overdue < 3)
    standard_case.days_overdue = 1
    standard_case.amount_at_risk = 2000.0
    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.ESCALATE not in decision.allowed_actions
    assert "escalation_aging_insufficient" in decision.prohibited_actions[ActionType.ESCALATE]

    # Case amount too small (amount_at_risk < 500)
    standard_case.days_overdue = 10
    standard_case.amount_at_risk = 50.0
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.ESCALATE not in decision2.allowed_actions
    assert "escalation_amount_insufficient" in decision2.prohibited_actions[ActionType.ESCALATE]


# 6. Customer Consent & Communication Semantics
def test_communication_consent_semantics(standard_case, standard_customer):
    """8. Test explicit consent semantics: either email or SMS opt-in allows communication."""
    engine = PolicyEngine()
    
    # Neither email nor SMS opt-in -> prohibited
    standard_customer.opt_in_email = False
    standard_customer.opt_in_sms = False
    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER not in decision.allowed_actions
    assert ActionType.PAYMENT_UPDATE not in decision.allowed_actions
    assert "consent_missing" in decision.prohibited_actions[ActionType.REMINDER]
    assert "consent_missing" in decision.prohibited_actions[ActionType.PAYMENT_UPDATE]

    # Email only opt-in -> allowed
    standard_customer.opt_in_email = True
    standard_customer.opt_in_sms = False
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER in decision2.allowed_actions
    assert ActionType.PAYMENT_UPDATE in decision2.allowed_actions

    # SMS only opt-in -> allowed
    standard_customer.opt_in_email = False
    standard_customer.opt_in_sms = True
    decision3 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER in decision3.allowed_actions
    assert ActionType.PAYMENT_UPDATE in decision3.allowed_actions


# 7. Contact Frequency Limits
def test_contact_frequency_limits(standard_case, standard_customer):
    """9. Test per-case reminder limit and global customer contact cap."""
    engine = PolicyEngine(PolicyConfig(
        max_reminders_per_case=2,
        max_total_contacts_per_customer=5,
    ))

    # Case reminder limit reached (reminder_count = 2)
    standard_case.reminder_count = 2
    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER not in decision.allowed_actions
    assert "reminder_limit_exceeded" in decision.prohibited_actions[ActionType.REMINDER]

    # Global customer contact limit reached (previous_contact_count = 5)
    standard_case.reminder_count = 0
    standard_customer.previous_contact_count = 5
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.REMINDER not in decision2.allowed_actions
    assert ActionType.PAYMENT_UPDATE not in decision2.allowed_actions
    assert "global_contact_limit_exceeded" in decision2.prohibited_actions[ActionType.REMINDER]
    assert "payment_update_contact_limit" in decision2.prohibited_actions[ActionType.PAYMENT_UPDATE]


# 8. Wait limits
def test_wait_limit_reached(standard_case, standard_customer):
    """10. Test WAIT limit reached after max consecutive days."""
    engine = PolicyEngine(PolicyConfig(max_consecutive_wait_days=14))
    standard_case.days_waiting = 14

    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.WAIT not in decision.allowed_actions
    assert "wait_limit_exceeded" in decision.prohibited_actions[ActionType.WAIT]


# 9. STOP action & terminal state handling
def test_stop_action_eligibility_and_case_state(standard_case, standard_customer):
    """11. Test STOP remains allowed in active case, but closed cases reject non-STOP actions."""
    engine = PolicyEngine()
    
    # Active case -> STOP is allowed along with active actions
    decision = engine.evaluate(standard_case, standard_customer)
    assert ActionType.STOP in decision.allowed_actions

    # Resolved recovered case -> all active recovery actions blocked
    standard_case.state = CaseState.RESOLVED_RECOVERED
    decision2 = engine.evaluate(standard_case, standard_customer)
    assert ActionType.RETRY not in decision2.allowed_actions
    assert ActionType.REMINDER not in decision2.allowed_actions
    assert ActionType.STOP in decision2.allowed_actions


# 10. Simultaneous multi-action prohibitions
def test_multiple_actions_prohibited_simultaneously(standard_case, standard_customer):
    """12. Test multiple simultaneous prohibitions leaving only STOP allowed."""
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

    assert ActionType.RETRY in decision.prohibited_actions
    assert ActionType.WAIT in decision.prohibited_actions
    assert ActionType.ESCALATE in decision.prohibited_actions
    assert ActionType.REMINDER in decision.prohibited_actions
    assert ActionType.PAYMENT_UPDATE in decision.prohibited_actions
    assert decision.allowed_actions == [ActionType.STOP]


# 11. Determinism and version propagation
def test_determinism_identical_inputs(standard_case, standard_customer):
    """13. Test identical inputs produce bit-identical allowed and prohibited actions."""
    engine = PolicyEngine()
    dec1 = engine.evaluate(standard_case, standard_customer)
    dec2 = engine.evaluate(standard_case, standard_customer)

    assert dec1.allowed_actions == dec2.allowed_actions
    assert dec1.prohibited_actions == dec2.prohibited_actions
    assert dec1.policy_version == dec2.policy_version


def test_policy_version_propagation(standard_case, standard_customer):
    """14. Test policy version string propagation."""
    engine = PolicyEngine(PolicyConfig(policy_version="policy_v5.2.1-custom"))
    decision = engine.evaluate(standard_case, standard_customer)
    assert decision.policy_version == "policy_v5.2.1-custom"
