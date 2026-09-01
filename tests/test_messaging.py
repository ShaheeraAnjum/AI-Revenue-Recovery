"""Comprehensive unit tests for Constrained LLM Messaging and Deterministic Message Validator covering all 23 Phase 9 criteria."""
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
    CommunicationChannel,
)
from src.messaging.templates import APPROVED_TEMPLATES, MessageTemplate
from src.messaging.validator import MessageValidator
from src.messaging.service import LLMMessageService


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


# 1. Correct failure code in body -> APPROVED
def test_correct_failure_code_in_body_approved(msg_case, msg_customer):
    """1. Verify message containing canonical ground truth failure code phrase in body is APPROVED."""
    service = LLMMessageService()
    msg, val_res, audit = service.generate_and_validate(
        case=msg_case,
        customer=msg_customer,
        selected_action=ActionType.PAYMENT_UPDATE,
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        decision_id="DEC-MSG-01",
    )
    assert val_res.is_approved is True
    assert val_res.status == MessageValidationStatus.APPROVED
    assert msg is not None
    assert "insufficient funds" in msg.body_text


# 2. Wrong failure code in body -> REJECTED
def test_wrong_failure_code_in_body_rejected(msg_case, msg_customer):
    """2. Verify message containing wrong failure code in body is REJECTED."""
    validator = MessageValidator()
    hallucinated_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 1500.00 for case CASE-MSG-1 could not be processed due to card expired. Please update your payment method.",
        stated_amount=1500.0,
        stated_failure_reason="CARD_EXPIRED",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(hallucinated_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.FAILURE_CODE_MISMATCH in res.rejection_reasons


# 3. Generic "declined" cannot satisfy specific failure code like INSUFFICIENT_FUNDS
def test_generic_declined_cannot_satisfy_specific_failure_code(msg_case, msg_customer):
    """3. Verify generic term 'declined' fails to satisfy specific ground truth code INSUFFICIENT_FUNDS."""
    validator = MessageValidator()
    generic_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 1500.00 for case CASE-MSG-1 could not be processed due to card declined. Please update your payment method.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(generic_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.FAILURE_CODE_MISMATCH in res.rejection_reasons


# 4. Correct amount in body -> APPROVED
def test_correct_amount_in_body_approved(msg_case, msg_customer):
    """4. Verify exact ground truth amount in body is APPROVED."""
    validator = MessageValidator()
    valid_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 1500.00 for case CASE-MSG-1 could not be processed due to insufficient funds. Please update your payment method.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(valid_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is True


# 5. Wrong amount in body -> REJECTED
def test_wrong_amount_in_body_rejected(msg_case, msg_customer):
    """5. Verify wrong amount in body (e.g. 5000.00 vs 1500.00) is REJECTED."""
    validator = MessageValidator()
    wrong_amount_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 5000.00 for case CASE-MSG-1 could not be processed due to insufficient funds. Please update your payment method.",
        stated_amount=5000.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(wrong_amount_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.AMOUNT_MISMATCH in res.rejection_reasons


# 6. Additional unsupported monetary amount -> REJECTED
def test_additional_unsupported_monetary_amount_rejected(msg_case, msg_customer):
    """6. Verify additional unapproved monetary amount in body is REJECTED."""
    validator = MessageValidator()
    extra_amount_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 1500.00 for case CASE-MSG-1 failed. Plus a fee of 50.00 applies. Due to insufficient funds. Please update your payment method.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(extra_amount_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.AMOUNT_MISMATCH in res.rejection_reasons


# 7. Structured amount correct but body amount wrong -> REJECTED
def test_structured_amount_correct_but_body_wrong_rejected(msg_case, msg_customer):
    """7. Verify structured stated_amount=1500 cannot mask a hallucinated body amount of 3500."""
    validator = MessageValidator()
    sneaky_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 3500.00 for case CASE-MSG-1 could not be processed due to insufficient funds. Please update your payment method.",
        stated_amount=1500.0,  # Structured metadata says 1500.0, but body says 3500.00!
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(sneaky_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.AMOUNT_MISMATCH in res.rejection_reasons


# 8. Structured failure reason correct but body reason wrong -> REJECTED
def test_structured_reason_correct_but_body_wrong_rejected(msg_case, msg_customer):
    """8. Verify structured stated_failure_reason cannot mask a hallucinated body reason."""
    validator = MessageValidator()
    sneaky_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 1500.00 for case CASE-MSG-1 could not be processed due to card expired. Please update your payment method.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",  # Structured is correct, but body is hallucinated!
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(sneaky_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.FAILURE_CODE_MISMATCH in res.rejection_reasons


# 9 & 10. Selected action mismatch; LLM cannot change selected_action
def test_selected_action_mismatch_rejected_and_immutable(msg_case, msg_customer):
    """9, 10, 21. Verify action mismatch is REJECTED and selected_action remains immutable."""
    service = LLMMessageService()
    decision_action = ActionType.PAYMENT_UPDATE

    rogue_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="We are retrying your scheduled payment of 1500.00 for case CASE-MSG-1.",
        stated_action=ActionType.RETRY,
    )

    delivered, val_res, audit = service.generate_and_validate(
        case=msg_case,
        customer=msg_customer,
        selected_action=decision_action,
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        decision_id="DEC-ACTION-01",
        override_candidate_message=rogue_msg,
    )

    assert val_res.is_approved is False
    assert MessageRejectionReason.ACTION_MISMATCH in val_res.rejection_reasons
    assert delivered is None
    assert audit.selected_action == ActionType.PAYMENT_UPDATE


# 11, 12, 13. Channel-specific consent checks
def test_channel_specific_consent_enforcement(msg_case, msg_customer):
    """11, 12, 13. Verify email consent for email templates and SMS consent for SMS templates."""
    validator = MessageValidator()

    # Case A: Email template with opt_in_email=False -> REJECTED
    cust_no_email = msg_customer.model_copy(deep=True)
    cust_no_email.opt_in_email = False
    cust_no_email.opt_in_sms = True  # Has SMS, but lacks EMAIL!

    msg_email = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 1500.00 for case CASE-MSG-1 could not be processed due to insufficient funds. Please update your payment method.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res_email = validator.validate(msg_email, msg_case, cust_no_email, ActionType.PAYMENT_UPDATE)
    assert res_email.is_approved is False
    assert MessageRejectionReason.CONSENT_MISSING in res_email.rejection_reasons

    # Case B: SMS template with opt_in_sms=False -> REJECTED
    cust_no_sms = msg_customer.model_copy(deep=True)
    cust_no_sms.opt_in_email = True
    cust_no_sms.opt_in_sms = False

    msg_sms = CandidateMessage(
        template_id="TMPL_REMINDER_SMS_V1",
        body_text="Reminder: Payment of 1500.00 for CASE-MSG-1 is due. Please check your account.",
        stated_amount=1500.0,
        stated_action=ActionType.REMINDER,
    )
    res_sms = validator.validate(msg_sms, msg_case, cust_no_sms, ActionType.REMINDER)
    assert res_sms.is_approved is False
    assert MessageRejectionReason.CONSENT_MISSING in res_sms.rejection_reasons


# 14 & 15. Unapproved template & template/action mismatch
def test_unapproved_template_and_action_mismatch(msg_case, msg_customer):
    """14, 15. Verify unapproved template and action mismatch are REJECTED."""
    validator = MessageValidator()
    
    # Unapproved template
    msg_unauth = CandidateMessage(
        template_id="TMPL_UNKNOWN_CUSTOM",
        body_text="Your payment of 1500.00 for case CASE-MSG-1 failed.",
    )
    res_unauth = validator.validate(msg_unauth, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res_unauth.is_approved is False
    assert MessageRejectionReason.UNAPPROVED_TEMPLATE in res_unauth.rejection_reasons

    # Template applied to wrong action (e.g. RETRY template used for ESCALATE)
    msg_mismatch = CandidateMessage(
        template_id="TMPL_RETRY_NOTICE_EMAIL_V1",
        body_text="Notice: We are re-attempting your scheduled payment of 1500.00 for case CASE-MSG-1.",
    )
    res_mismatch = validator.validate(msg_mismatch, msg_case, msg_customer, ActionType.ESCALATE)
    assert res_mismatch.is_approved is False
    assert MessageRejectionReason.ACTION_MISMATCH in res_mismatch.rejection_reasons


# 16. Unsupported/invented URL -> REJECTED
def test_unsupported_invented_url_rejected(msg_case, msg_customer):
    """16. Verify LLM inventing external URLs is REJECTED with FORBIDDEN_CONTENT."""
    validator = MessageValidator()
    url_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 1500.00 for case CASE-MSG-1 could not be processed due to insufficient funds. Please update your payment method at http://fake-phishing-url.com.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(url_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.FORBIDDEN_CONTENT in res.rejection_reasons


# 17. Missing required factual field in body -> REJECTED
def test_missing_required_case_id_rejected(msg_case, msg_customer):
    """17. Verify missing required factual field (case_id) in body is REJECTED."""
    validator = MessageValidator()
    missing_id_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Dear customer, your payment of 1500.00 could not be processed due to insufficient funds. Please update your payment method.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.PAYMENT_UPDATE,
    )
    res = validator.validate(missing_id_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)
    assert res.is_approved is False
    assert MessageRejectionReason.REQUIRED_FACT_MISSING in res.rejection_reasons


# 18 & 19. Repeated identical validation and offline operation
def test_repeated_validation_offline_determinism(msg_case, msg_customer):
    """18, 19. Verify repeated validation runs deterministically offline without network calls."""
    validator = MessageValidator()
    valid_msg = CandidateMessage(
        template_id="TMPL_REMINDER_EMAIL_V1",
        body_text="Reminder: Payment of 1500.00 for reference CASE-MSG-1 is overdue due to insufficient funds. Please review your billing details.",
        stated_amount=1500.0,
        stated_failure_reason="INSUFFICIENT_FUNDS",
        stated_action=ActionType.REMINDER,
    )
    res1 = validator.validate(valid_msg, msg_case, msg_customer, ActionType.REMINDER)
    res2 = validator.validate(valid_msg, msg_case, msg_customer, ActionType.REMINDER)
    assert res1.is_approved is True
    assert res2.is_approved is True
    assert res1.rejection_reasons == res2.rejection_reasons


# 20, 22, 23. Ground truth immutability
def test_ground_truth_remains_unmodified(msg_case, msg_customer):
    """20, 22, 23. Verify validator never modifies case amount, failure_code, or customer state."""
    validator = MessageValidator()
    bad_msg = CandidateMessage(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        body_text="Payment of 9999.00 for case CASE-MSG-1 failed due to card expired.",
        stated_amount=9999.0,
        stated_failure_reason="CARD_EXPIRED",
    )
    validator.validate(bad_msg, msg_case, msg_customer, ActionType.PAYMENT_UPDATE)

    # Invariants
    assert msg_case.amount_at_risk == 1500.0
    assert msg_case.failure_code == PaymentFailureCode.INSUFFICIENT_FUNDS
    assert msg_case.case_id == "CASE-MSG-1"
    assert msg_customer.customer_id == "CUST-MSG-1"
