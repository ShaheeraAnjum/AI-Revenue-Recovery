"""Strict deterministic validator protecting against LLM hallucination and policy violations."""
import re
from typing import Optional, List, Dict
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile, PaymentFailureCode
from src.messaging.schema import (
    CandidateMessage,
    ValidationResult,
    MessageValidationStatus,
    MessageRejectionReason,
    CommunicationChannel,
    DEFAULT_VALIDATOR_VERSION,
)
from src.messaging.templates import MessageTemplate, APPROVED_TEMPLATES

# Canonical approved phrases per failure code (strictly mapped; generic terms cannot substitute specific codes)
CANONICAL_FAILURE_PHRASES: Dict[PaymentFailureCode, List[str]] = {
    PaymentFailureCode.INSUFFICIENT_FUNDS: ["insufficient funds"],
    PaymentFailureCode.CARD_EXPIRED: ["card expired", "expired card"],
    PaymentFailureCode.DO_NOT_HONOR: ["do not honor", "bank declined charge"],
    PaymentFailureCode.PROCESSING_ERROR: ["processing error"],
    PaymentFailureCode.FRAUD_SUSPECTED: ["suspected fraud", "security flag"],
    PaymentFailureCode.INVALID_CARD_NUMBER: ["invalid card number"],
    PaymentFailureCode.AUTHENTICATION_REQUIRED: ["authentication required", "3d secure required"],
    PaymentFailureCode.LIMIT_EXCEEDED: ["limit exceeded", "spending limit reached"],
    PaymentFailureCode.GENERIC_DECLINE: ["card declined", "transaction declined", "declined"],
}

# Action action-intent keywords in message body to detect action mismatch
ACTION_INTENT_KEYWORDS: Dict[ActionType, List[str]] = {
    ActionType.RETRY: ["re-attempting", "retrying your scheduled payment", "re-trying"],
    ActionType.PAYMENT_UPDATE: ["update your payment method", "updating your payment"],
    ActionType.REMINDER: ["reminder", "is overdue", "review your billing details"],
}


class MessageValidator:
    """Deterministic message validator ensuring zero LLM hallucination and 100% ground-truth adherence."""

    def __init__(self, templates: Optional[Dict[str, MessageTemplate]] = None, version: str = DEFAULT_VALIDATOR_VERSION):
        self.templates = templates or APPROVED_TEMPLATES
        self.version = version

    def validate(
        self,
        message: CandidateMessage,
        case: RecoveryCase,
        customer: CustomerProfile,
        selected_action: ActionType,
        template: Optional[MessageTemplate] = None,
    ) -> ValidationResult:
        """Deterministically validate candidate message body against authoritative ground truth facts."""
        reasons: List[MessageRejectionReason] = []

        # 1. Template approval check
        active_template = template or self.templates.get(message.template_id)
        if active_template is None or active_template.template_id not in self.templates:
            reasons.append(MessageRejectionReason.UNAPPROVED_TEMPLATE)
            return ValidationResult(
                is_approved=False,
                status=MessageValidationStatus.REJECTED,
                rejection_reasons=reasons,
                validator_version=self.version,
            )

        # 2. Template / Action compatibility check
        if selected_action not in active_template.applicable_actions:
            reasons.append(MessageRejectionReason.ACTION_MISMATCH)

        if message.stated_action is not None and message.stated_action != selected_action:
            reasons.append(MessageRejectionReason.ACTION_MISMATCH)

        # Body action intent check
        body_lower = message.body_text.lower()
        for action_type, keywords in ACTION_INTENT_KEYWORDS.items():
            if action_type != selected_action:
                for kw in keywords:
                    if kw in body_lower:
                        reasons.append(MessageRejectionReason.ACTION_MISMATCH)
                        break

        # 3. Channel-specific consent check
        if active_template.requires_consent:
            if active_template.channel == CommunicationChannel.EMAIL:
                if not customer.opt_in_email:
                    reasons.append(MessageRejectionReason.CONSENT_MISSING)
            elif active_template.channel == CommunicationChannel.SMS:
                if not customer.opt_in_sms:
                    reasons.append(MessageRejectionReason.CONSENT_MISSING)

        # 4. Anti-Hallucination Amount Validation on Body Text
        expected_amount = float(case.amount_at_risk)
        # Remove case_id to prevent capturing numbers inside ID strings like CASE-123
        body_clean = message.body_text.replace(case.case_id, "")
        
        # Check for unapproved external URLs
        if re.search(r"https?://", body_clean):
            reasons.append(MessageRejectionReason.FORBIDDEN_CONTENT)

        # Find all monetary numbers in body
        found_numbers = re.findall(r"\b\d+(?:\.\d{1,2})?\b", body_clean)

        if "amount" in active_template.required_facts:
            if not found_numbers:
                reasons.append(MessageRejectionReason.AMOUNT_MISSING)
            else:
                # Check if correct ground truth amount is present
                matching_amount = any(abs(float(num) - expected_amount) < 0.01 for num in found_numbers)
                if not matching_amount:
                    reasons.append(MessageRejectionReason.AMOUNT_MISMATCH)
                
                # Check if there are hallucinated additional monetary numbers
                for num in found_numbers:
                    if abs(float(num) - expected_amount) >= 0.01:
                        reasons.append(MessageRejectionReason.AMOUNT_MISMATCH)
                        break
        
        # Check structured stated_amount
        if message.stated_amount is not None:
            if abs(message.stated_amount - expected_amount) > 0.01:
                if MessageRejectionReason.AMOUNT_MISMATCH not in reasons:
                    reasons.append(MessageRejectionReason.AMOUNT_MISMATCH)

        # 5. Anti-Hallucination Failure Code Validation on Body Text
        if "failure_reason" in active_template.required_facts:
            actual_code = case.failure_code
            approved_phrases = CANONICAL_FAILURE_PHRASES.get(actual_code, [])

            # Body text must contain at least one approved canonical phrase for the actual code
            has_actual_phrase = any(phrase in body_lower for phrase in approved_phrases)
            if not has_actual_phrase:
                reasons.append(MessageRejectionReason.FAILURE_CODE_MISMATCH)

            # Body text must NOT contain canonical phrases of other failure codes
            for other_code, other_phrases in CANONICAL_FAILURE_PHRASES.items():
                if other_code != actual_code:
                    for op in other_phrases:
                        if op in body_lower:
                            if MessageRejectionReason.FAILURE_CODE_MISMATCH not in reasons:
                                reasons.append(MessageRejectionReason.FAILURE_CODE_MISMATCH)
                            break

            # Structured stated_failure_reason check
            if message.stated_failure_reason is not None:
                if message.stated_failure_reason != actual_code.value:
                    if MessageRejectionReason.FAILURE_CODE_MISMATCH not in reasons:
                        reasons.append(MessageRejectionReason.FAILURE_CODE_MISMATCH)

        # 6. Required Facts in Body: Case ID
        if "case_id" in active_template.required_facts:
            if case.case_id not in message.body_text:
                reasons.append(MessageRejectionReason.REQUIRED_FACT_MISSING)

        is_approved = (len(reasons) == 0)
        return ValidationResult(
            is_approved=is_approved,
            status=MessageValidationStatus.APPROVED if is_approved else MessageValidationStatus.REJECTED,
            rejection_reasons=reasons,
            validator_version=self.version,
        )
