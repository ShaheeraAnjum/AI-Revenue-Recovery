"""Comprehensive unit tests for Constrained LLM Messaging and Deterministic Message Validator."""
import pytest
from src.domain.actions import ActionType
from src.domain.case import (
    RecoveryCase,
    CustomerProfile,
    PaymentFailureCode,
    PaymentMethodType,
    CaseState,
)
from src.messaging.schema import (
    CandidateMessage,
    MessageValidationStatus,
    MessageRejectionReason,
)
from src.messaging.templates import APPROVED_TEMPLATES, MessageTemplate
from src.messaging.validator import MessageValidator
from src.messaging.service import LLMMessageService
from src.messaging.llm_client import MockLLMClient


@pytest.fixture
def msg_customer() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-MSG-1",
        customer_value=5000.0,
        subscription_age_days=100,
        previous_success_rate=0.8,
        previous_contact_count=0,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
        opt_in_email=True,
        opt_in_sms=True,
    )


@pytest.fixture
def msg_case() -> RecoveryCase:
    return RecoveryCase(
        case_id="CASE-MSG-1",
        customer_id="CUST-MSG-1",
        amount_at_risk=1500.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=2,
        days_waiting=0,
        state=CaseState.ACTIVE,
    )


# 1 & 3 & 6. Correct failure code, amount, and action -> APPROVED
def test_valid_message_generation_and_approval(msg_case, msg_customer):
    """1, 3, 6. Verify valid candidate message matching all ground truth facts is APPROVED."""
    service = LLMMessageService()
    msg, val_res, audit = service.generate_and_validate(
        case=msg_case,
        customer=msg_customer,
        selected_action=ActionType.PAYMENT_UPDATE,
        template_id="TMPL_PAYMENT_UPDATE_V1",
        decision_id="DEC-MSG-01",
    )

    assert val_res.is_approved is True
    assert val_res.status == MessageValidationStatus.APPROVED
    assert msg is not None
    assert "1500.00" in msg.body_text
    assert "insufficient funds" in msg.body_text
    assert audit.validation_status == MessageValidationStatus.APPROVED


# 2. Wrong failure code hallucination -> REJECTED
def test_wrong_failure_code_hallucination_rejected(msg_case, msg_customer):
    """2. Verify LLM hallucinating CARD_EXPIRED when actual is INSUFFICIENT_FUNDS is REJECTED."""
    validator = MessageValidator()
    hallucinated_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_V1",
        body_text="Your card expired for case CASE-MSG-1 with amount 1500.00. Please update.",
        stated_amount=1500.0,
        stated_failure_reason="CARD_EXPIRED",
        stated_action=ActionType.PAYMENT_UPDATE,
    )

    res = validator.validate(hallucinated_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.FAILURE_CODE_MISMATCH in res.rejection_reasons


# 4. Wrong amount hallucination -> REJECTED
def test_wrong_amount_hallucination_rejected(msg_case, msg_customer):
    """4. Verify LLM claiming 5000 when actual is 1500 is REJECTED."""
    validator = MessageValidator()
    hallucinated_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_V1",
        body_text="Please pay 5000.00 for case CASE-MSG-1 due to insufficient funds.",
        stated_amount=5000.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )

    res = validator.validate(hallucinated_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.AMOUNT_MISMATCH in res.rejection_reasons


# 5. Missing amount -> REJECTED
def test_missing_amount_rejected(msg_case, msg_customer):
    """5. Verify message omitting required amount is REJECTED."""
    validator = MessageValidator()
    msg_without_amount = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_V1",
        body_text="Payment failed for case CASE-MSG-1 due to insufficient funds.",
        stated_amount=None,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )

    res = validator.validate(msg_without_amount, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.AMOUNT_MISSING in res.rejection_reasons


# 7. LLM suggesting different action cannot change selected_action
def test_llm_cannot_alter_selected_action(msg_case, msg_customer):
    """7 & 14. Verify LLM suggesting RETRY when Decision Engine chose PAYMENT_UPDATE cannot change decision."""
    service = LLMMessageService()
    decision_action = ActionType.PAYMENT_UPDATE

    rogue_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_V1",
        body_text="We are retrying your charge of 1500.00 for case CASE-MSG-1.",
        stated_action=ActionType.RETRY,
    )

    delivered, val_res, audit = service.generate_and_validate(
        case=msg_case,
        customer=msg_customer,
        selected_action=decision_action,
        template_id="TMPL_PAYMENT_UPDATE_V1",
        decision_id="DEC-ROGUE-01",
        override_candidate_message=rogue_msg,
    )

    assert val_res.is_approved is False
    assert MessageRejectionReason.ACTION_MISMATCH in val_res.rejection_reasons
    assert delivered is None
    assert audit.selected_action == ActionType.PAYMENT_UPDATE


# 8. Missing consent -> REJECTED
def test_missing_consent_rejected(msg_case, msg_customer):
    """8. Verify customer with opt_in_email=False and opt_in_sms=False is REJECTED."""
    no_consent_cust = msg_customer.model_copy(deep=True)
    no_consent_cust.opt_in_email = False
    no_consent_cust.opt_in_sms = False

    service = LLMMessageService()
    msg, val_res, audit = service.generate_and_validate(
        case=msg_case,
        customer=no_consent_cust,
        selected_action=ActionType.PAYMENT_UPDATE,
        template_id="TMPL_PAYMENT_UPDATE_V1",
        decision_id="DEC-CONSENT-01",
    )

    assert val_res.is_approved is False
    assert MessageRejectionReason.CONSENT_MISSING in val_res.rejection_reasons
    assert msg is None


# 9. Unapproved template -> REJECTED
def test_unapproved_template_rejected(msg_case, msg_customer):
    """9. Verify unauthorized template ID is REJECTED."""
    service = LLMMessageService()
    msg, val_res, audit = service.generate_and_validate(
        case=msg_case,
        customer=msg_customer,
        selected_action=ActionType.PAYMENT_UPDATE,
        template_id="TMPL_UNAUTHORIZED_HACK",
        decision_id="DEC-UNAUTH-01",
    )

    assert val_res.is_approved is False
    assert msg is None


# 10. Missing case_id fact -> REJECTED
def test_missing_case_id_rejected(msg_case, msg_customer):
    """10. Verify message missing case_id is REJECTED."""
    validator = MessageValidator()
    msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_V1",
        body_text="Your payment of 1500.00 failed due to insufficient funds.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )

    res = validator.validate(msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.REQUIRED_FACT_MISSING in res.rejection_reasons


# 12 & 13. Deterministic repeated validation offline without network access
def test_deterministic_offline_validation(msg_case, msg_customer):
    """12 & 13. Verify validator operates deterministically offline without network calls."""
    validator = MessageValidator()
    valid_msg = CandidateMessage(
        template_id="TMPL_REMINDER_V1",
        body_text="Reminder: Payment of 1500.00 for reference CASE-MSG-1 is overdue due to insufficient funds.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.REMINDER,
    )

    res1 = validator.validate(valid_msg, msg_case, msg_customer, ActionType.REMINDER)
    res2 = validator.validate(valid_msg, msg_case, msg_customer, ActionType.REMINDER)

    assert res1.is_approved == res2.is_approved
    assert res1.rejection_reasons == res2.rejection_reasons
