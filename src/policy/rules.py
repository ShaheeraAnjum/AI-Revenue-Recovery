"""Deterministic rule definitions and reason codes for policy validation."""
from enum import Enum
from typing import Optional, Tuple
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile, CaseState
from src.policy.config import PolicyConfig


class PolicyRejectionReason(str, Enum):
    """Explicit machine-readable rejection reasons."""
    # Retry rejections
    RETRY_LIMIT_EXCEEDED = "retry_limit_exceeded: maximum allowed retries reached for this case"
    NETWORK_RULE_HARD_DECLINE = "network_rule_hard_decline: card network rules prohibit retrying this failure code"
    CASE_NOT_ACTIVE = "case_not_active: case is not in an active/detected recovery state"

    # Reminder rejections
    CONSENT_MISSING = "consent_missing: customer has not opted into email or sms communications"
    REMINDER_LIMIT_EXCEEDED = "reminder_limit_exceeded: maximum reminders reached for this case"
    GLOBAL_CONTACT_LIMIT_EXCEEDED = "global_contact_limit_exceeded: customer contact frequency cap reached"

    # Payment update rejections
    PAYMENT_UPDATE_CONTACT_LIMIT = "payment_update_contact_limit: customer contact cap prevents sending update link"
    PAYMENT_UPDATE_CONSENT_MISSING = "payment_update_consent_missing: no valid communication channel for update request"

    # Wait rejections
    WAIT_LIMIT_EXCEEDED = "wait_limit_exceeded: maximum consecutive wait days reached"

    # Escalation rejections
    ESCALATION_LIMIT_EXCEEDED = "escalation_limit_exceeded: case has already reached maximum escalation limit"
    ESCALATION_AGING_INSUFFICIENT = "escalation_aging_insufficient: days overdue below minimum threshold for human intervention"
    ESCALATION_AMOUNT_INSUFFICIENT = "escalation_amount_insufficient: amount at risk below minimum threshold for escalation"

    # Stop rejections
    CASE_ALREADY_RESOLVED = "case_already_resolved: case is already in a terminal resolved state"

    # PCI Boundary
    PCI_COMPLIANCE_RESTRICTION = "pci_compliance_restriction: direct capture of raw payment credentials prohibited"


def check_action_compliance(
    action: ActionType,
    case: RecoveryCase,
    customer: CustomerProfile,
    config: PolicyConfig,
) -> Tuple[bool, Optional[str]]:
    """Evaluate compliance constraints for a candidate action.
    Returns (is_allowed, rejection_reason_or_none).
    """
    # 0. Case lifecycle check
    if case.state in {CaseState.RESOLVED_RECOVERED, CaseState.RESOLVED_UNRECOVERABLE}:
        if action != ActionType.STOP:
            return False, PolicyRejectionReason.CASE_NOT_ACTIVE.value

    # 1. RETRY action checks
    if action == ActionType.RETRY:
        if case.retry_attempt_count >= config.max_retries_per_case:
            return False, PolicyRejectionReason.RETRY_LIMIT_EXCEEDED.value
        if case.failure_code in config.prohibited_retry_failure_codes:
            return False, PolicyRejectionReason.NETWORK_RULE_HARD_DECLINE.value
        return True, None

    # 2. REMINDER action checks
    elif action == ActionType.REMINDER:
        if config.require_explicit_consent_for_reminder and not (customer.opt_in_email or customer.opt_in_sms):
            return False, PolicyRejectionReason.CONSENT_MISSING.value
        if case.reminder_count >= config.max_reminders_per_case:
            return False, PolicyRejectionReason.REMINDER_LIMIT_EXCEEDED.value
        if customer.previous_contact_count >= config.max_total_contacts_per_customer:
            return False, PolicyRejectionReason.GLOBAL_CONTACT_LIMIT_EXCEEDED.value
        return True, None

    # 3. PAYMENT_UPDATE action checks
    elif action == ActionType.PAYMENT_UPDATE:
        if config.require_consent_for_payment_update and not (customer.opt_in_email or customer.opt_in_sms):
            return False, PolicyRejectionReason.PAYMENT_UPDATE_CONSENT_MISSING.value
        if customer.previous_contact_count >= config.max_total_contacts_per_customer:
            return False, PolicyRejectionReason.PAYMENT_UPDATE_CONTACT_LIMIT.value
        return True, None

    # 4. WAIT action checks
    elif action == ActionType.WAIT:
        if case.days_waiting >= config.max_consecutive_wait_days:
            return False, PolicyRejectionReason.WAIT_LIMIT_EXCEEDED.value
        return True, None

    # 5. ESCALATE action checks
    elif action == ActionType.ESCALATE:
        # Check aging threshold
        if case.days_overdue < config.min_days_overdue_for_escalation:
            return False, PolicyRejectionReason.ESCALATION_AGING_INSUFFICIENT.value
        # Check minimum amount threshold
        if case.amount_at_risk < config.min_amount_for_escalation:
            return False, PolicyRejectionReason.ESCALATION_AMOUNT_INSUFFICIENT.value
        return True, None

    # 6. STOP action checks
    elif action == ActionType.STOP:
        # STOP is always eligible during active recovery to terminate unrecoverable loops
        return True, None

    return False, "unknown_action_type"
