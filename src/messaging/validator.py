"""Strict deterministic validator protecting against LLM hallucination and policy violations."""
import re
from typing import Optional, List
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile, PaymentFailureCode
from src.messaging.schema import (
    CandidateMessage,
    ValidationResult,
    MessageValidationStatus,
    MessageRejectionReason,
    DEFAULT_VALIDATOR_VERSION,
)
from src.messaging.templates import MessageTemplate, APPROVED_TEMPLATES

# Failure code human-readable token mapping
FAILURE_CODE_KEYWORDS = {
    PaymentFailureCode.INSUFFICIENT_FUNDS: ["insufficient funds", "insufficient_funds", "low balance"],
    PaymentFailureCode.CARD_EXPIRED: ["card expired", "card_expired", "expiration"],
    PaymentFailureCode.DO_NOT_HONOR: ["do not honor", "do_not_honor", "bank decline"],
    PaymentFailureCode.FRAUD_SUSPECTED: ["fraud suspected", "fraud", "security block"],
    PaymentFailureCode.INVALID_CARD_NUMBER: ["invalid card", "invalid card number"],
    PaymentFailureCode.AUTHENTICATION_REQUIRED: ["authentication required", "3ds", "verification required"],
    PaymentFailureCode.PROCESSING_ERROR: ["processing error", "system error"],
    PaymentFailureCode.GENERIC_DECLINE: ["generic decline", "declined", "card declined"],
    PaymentFailureCode.LIMIT_EXCEEDED: ["limit exceeded", "spending limit"],
}


class MessageValidator:
    """Deterministic message validator ensuring zero LLM hallucination."""

    def __init__(self, templates: Optional[dict[str, MessageTemplate]] = None, version: str = DEFAULT_VALIDATOR_VERSION):
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
        """Deterministically validate candidate message against authoritative ground truth facts."""
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

        # 2. Action consistency check
        if selected_action not in active_template.applicable_actions:
            reasons.append(MessageRejectionReason.ACTION_MISMATCH)

        if message.stated_action is not None and message.stated_action != selected_action:
            reasons.append(MessageRejectionReason.ACTION_MISMATCH)

        # 3. Consent check
        if active_template.requires_consent:
            has_consent = customer.opt_in_email or customer.opt_in_sms
            if not has_consent:
                reasons.append(MessageRejectionReason.CONSENT_MISSING)

        # 4. Amount consistency & anti-hallucination check
        expected_amount = float(case.amount_at_risk)
        if "amount" in active_template.required_facts:
            if message.stated_amount is None:
                # Remove case_id from body text before searching for amounts
                clean_body = message.body_text.replace(case.case_id, "")
                extracted_numbers = re.findall(r"\b\d+(?:\.\d{1,2})?\b", clean_body)
                if not extracted_numbers:
                    reasons.append(MessageRejectionReason.AMOUNT_MISSING)
                else:
                    found_match = any(abs(float(num) - expected_amount) < 0.01 for num in extracted_numbers)
                    if not found_match:
                        reasons.append(MessageRejectionReason.AMOUNT_MISMATCH)
            else:
                if abs(message.stated_amount - expected_amount) > 0.01:
                    reasons.append(MessageRejectionReason.AMOUNT_MISMATCH)

        # 5. Failure code consistency & anti-hallucination check
        if "failure_reason" in active_template.required_facts:
            actual_code = case.failure_code
            allowed_keywords = FAILURE_CODE_KEYWORDS.get(actual_code, [actual_code.value.lower()])

            # Verify stated failure reason if present
            if message.stated_failure_reason is not None:
                if not any(kw in message.stated_failure_reason.lower() for kw in allowed_keywords):
                    reasons.append(MessageRejectionReason.FAILURE_CODE_MISMATCH)

            # Check body text does not mention conflicting failure codes
            body_lower = message.body_text.lower()
            for other_code, other_kws in FAILURE_CODE_KEYWORDS.items():
                if other_code != actual_code:
                    for okw in other_kws:
                        if okw in body_lower and not any(akw in body_lower for akw in allowed_keywords):
                            reasons.append(MessageRejectionReason.FAILURE_CODE_MISMATCH)
                            break

        # 6. Case ID factual check
        if "case_id" in active_template.required_facts:
            if case.case_id not in message.body_text:
                reasons.append(MessageRejectionReason.REQUIRED_FACT_MISSING)

        is_valid = len(reasons) == 0
        return ValidationResult(
            is_approved=is_valid,
            status=MessageValidationStatus.APPROVED if is_valid else MessageValidationStatus.REJECTED,
            rejection_reasons=reasons,
            validator_version=self.version,
        )
