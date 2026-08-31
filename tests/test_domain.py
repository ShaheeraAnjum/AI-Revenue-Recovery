"""Unit tests for domain models, action classifications, and idempotency key."""
import pytest
from src.domain.actions import (
    ActionType,
    ActionCategory,
    is_supported_action,
    is_restricted_action,
    SUPPORTED_ACTIONS,
    RESTRICTED_ACTIONS,
    TERMINAL_ACTIONS,
)
from src.domain.case import (
    CaseState,
    PaymentFailureCode,
    PaymentMethodType,
    CustomerProfile,
    RecoveryCase,
    IdempotencyKey,
)
from src.domain.events import BillingEvent, PaymentFailureEvent
from src.domain.verification import VerificationStatus, GatewayExecutionResult


def test_action_classification():
    """Verify supported, restricted, and terminal action classifications."""
    assert is_supported_action(ActionType.RETRY) is True
    assert is_supported_action(ActionType.PAYMENT_UPDATE) is True
    assert is_supported_action(ActionType.REMINDER) is True
    assert is_supported_action(ActionType.WAIT) is True
    assert is_supported_action(ActionType.ESCALATE) is False
    assert is_supported_action(ActionType.STOP) is False

    assert is_restricted_action(ActionType.ESCALATE) is True
    assert is_restricted_action(ActionType.RETRY) is False
    assert is_restricted_action(ActionType.STOP) is False

    assert ActionType.STOP in TERMINAL_ACTIONS
    assert len(SUPPORTED_ACTIONS) == 4
    assert len(RESTRICTED_ACTIONS) == 1
    assert len(TERMINAL_ACTIONS) == 1


def test_idempotency_key_generation():
    """Verify exact 4-tuple structure (case_id, decision_id, action, attempt)."""
    key = IdempotencyKey(
        case_id="CASE-101",
        decision_id="DEC-202",
        action=ActionType.RETRY,
        attempt=1,
    )
    assert key.case_id == "CASE-101"
    assert key.decision_id == "DEC-202"
    assert key.action == ActionType.RETRY
    assert key.attempt == 1
    assert key.to_string() == "CASE-101:DEC-202:RETRY:1"


def test_customer_profile_and_case_validation():
    """Verify CustomerProfile and RecoveryCase schema constraints."""
    customer = CustomerProfile(
        customer_id="CUST-1",
        customer_value=12500.0,
        subscription_age_days=180,
        previous_success_rate=0.85,
        previous_contact_count=2,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
        opt_in_email=True,
        opt_in_sms=False,
        active_recovery_cases=1,
    )
    assert customer.customer_id == "CUST-1"
    assert customer.customer_value == 12500.0
    assert customer.audit_segment == "standard"

    case = RecoveryCase(
        case_id="CASE-1",
        customer_id="CUST-1",
        amount_at_risk=2500.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=3,
    )
    assert case.case_id == "CASE-1"
    assert case.state == CaseState.DETECTED
    assert case.amount_at_risk == 2500.0
    assert case.days_waiting == 0


def test_verification_status_enums():
    """Verify all 5 gateway verification statuses exist."""
    statuses = {s.value for s in VerificationStatus}
    assert statuses == {"SUCCESS", "DECLINE", "TIMEOUT", "UNKNOWN", "RECONCILIATION"}
