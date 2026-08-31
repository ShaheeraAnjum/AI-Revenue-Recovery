"""Deterministic rule definitions, reason codes, and compliance checks."""
from enum import Enum
from typing import Optional, Tuple
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile, CaseState
from src.policy.config import PolicyConfig


class PolicyRejectionReason(str, Enum):
    """Explicit machine-readable and human-understandable rejection reasons."""
    # Case state
    CASE_NOT_ACTIVE = "case_not_active: case is not in an active or detected recovery state"

    # Retry rejections
    RETRY_LIMIT_EXCEEDED = "retry_limit_exceeded: maximum allowed retries reached for this case"
    NETWORK_RULE_HARD_DECLINE = "network_rule_hard_decline: card network rules prohibit retrying this failure code"
    PCI_TOKENIZATION_VIOLATION = "pci_tokenization_boundary_violation: automated retry prohibited on non-tokenized payment credentials"
    VIP_EXPLORATION_PROTECTION = "vip_exploration_protection_active: high-value customer protected against repeated automated retries"

    # Reminder rejections
    CONSENT_MISSING = "consent_missing: customer has not opted into email or sms communications"
    REMINDER_LIMIT_EXCEEDED = "reminder_limit_exceeded: maximum reminders reached for this case"
    GLOBAL_CONTACT_LIMIT_EXCEEDED = "global_contact_limit_exceeded: customer contact frequency cap reached"

    # Payment update rejections
    PAYMENT_UPDATE_CONSENT_MISSING = "payment_update_consent_missing: customer has not opted into email or sms communication channel"
    PAYMENT_UPDATE_CONTACT_LIMIT = "payment_update_contact_limit: customer contact frequency cap prevents sending update request"

    # Wait rejections
    WAIT_LIMIT_EXCEEDED = "wait_limit_exceeded: maximum consecutive wait days reached"

    # Escalation rejections
    ESCALATION_LIMIT_EXCEEDED = "escalation_limit_exceeded: case has reached maximum allowed escalations"
    ESCALATION_AGING_INSUFFICIENT = "escalation_aging_insufficient: days overdue below minimum threshold for human intervention"
    ESCALATION_AMOUNT_INSUFFICIENT = "escalation_amount_insufficient: amount at risk below minimum threshold for human intervention"


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
        # A. Retry count check
        if case.retry_attempt_count >= config.max_retries_per_case:
            return False, PolicyRejectionReason.RETRY_LIMIT_EXCEEDED.value
        
        # B. Card network hard decline check
        if case.failure_code in config.prohibited_retry_failure_codes:
            return False, PolicyRejectionReason.NETWORK_RULE_HARD_DECLINE.value
        
        # C. PCI tokenization boundary check
        if config.enforce_pci_tokenization_boundary and not case.is_pci_tokenized:
            return False, PolicyRejectionReason.PCI_TOKENIZATION_VIOLATION.value
        
        # D. VIP exploration protection (prohibits repeated speculative retries on VIP accounts)
        if (
            config.enable_exploration_protection_for_vip
            and customer.customer_value >= config.vip_customer_value_threshold
            and case.retry_attempt_count > 0
        ):
            return False, PolicyRejectionReason.VIP_EXPLORATION_PROTECTION.value
        
        return True, None

    # 2. REMINDER action checks
    elif action == ActionType.REMINDER:
        # A. Consent check (either email OR SMS opt-in is required)
        has_consent = customer.opt_in_email or customer.opt_in_sms
        if config.require_explicit_consent_for_reminder and not has_consent:
            return False, PolicyRejectionReason.CONSENT_MISSING.value
        
        # B. Per-case reminder cap check
        if case.reminder_count >= config.max_reminders_per_case:
            return False, PolicyRejectionReason.REMINDER_LIMIT_EXCEEDED.value
        
        # C. Global customer contact cap check
        if customer.previous_contact_count >= config.max_total_contacts_per_customer:
            return False, PolicyRejectionReason.GLOBAL_CONTACT_LIMIT_EXCEEDED.value
        
        return True, None

    # 3. PAYMENT_UPDATE action checks
    elif action == ActionType.PAYMENT_UPDATE:
        # A. Consent/Contactability check (either email OR SMS opt-in is required to deliver the update link)
        has_channel = customer.opt_in_email or customer.opt_in_sms
        if config.require_consent_for_payment_update and not has_channel:
            return False, PolicyRejectionReason.PAYMENT_UPDATE_CONSENT_MISSING.value
        
        # B. Global customer contact cap check
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
        # A. Case escalation count limit check
        if case.escalation_count >= config.max_escalations_per_case:
            return False, PolicyRejectionReason.ESCALATION_LIMIT_EXCEEDED.value
        
        # B. Minimum overdue aging threshold check
        if case.days_overdue < config.min_days_overdue_for_escalation:
            return False, PolicyRejectionReason.ESCALATION_AGING_INSUFFICIENT.value
        
        # C. Minimum amount at risk threshold check
        if case.amount_at_risk < config.min_amount_for_escalation:
            return False, PolicyRejectionReason.ESCALATION_AMOUNT_INSUFFICIENT.value
        
        return True, None

    # 6. STOP action checks
    elif action == ActionType.STOP:
        # STOP is always eligible during active recovery to terminate unrecoverable loops
        return True, None

    return False, "unknown_action_type"
