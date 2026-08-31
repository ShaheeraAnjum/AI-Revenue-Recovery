"""Configuration-driven policy parameters and versioning."""
from typing import Dict, Set
from pydantic import BaseModel, Field
from src.domain.case import PaymentFailureCode

DEFAULT_POLICY_VERSION: str = "policy_v5.0.0"

# Card network hard decline codes that strictly prohibit automated retry per network compliance
HARD_DECLINE_CODES: Set[PaymentFailureCode] = {
    PaymentFailureCode.FRAUD_SUSPECTED,
    PaymentFailureCode.CARD_EXPIRED,
    PaymentFailureCode.INVALID_CARD_NUMBER,
    PaymentFailureCode.DO_NOT_HONOR,
}


class PolicyConfig(BaseModel):
    """Centralized, versioned configuration for safety and compliance constraints."""
    policy_version: str = DEFAULT_POLICY_VERSION

    # 1. Retry limits
    max_retries_per_case: int = Field(default=3, ge=1, description="Maximum automated gateway retry attempts per case")
    prohibited_retry_failure_codes: Set[PaymentFailureCode] = Field(
        default_factory=lambda: set(HARD_DECLINE_CODES),
        description="Failure codes where network rules strictly prohibit retry",
    )

    # 2. Communication and contact limits
    max_reminders_per_case: int = Field(default=2, ge=1, description="Max reminders per case")
    max_total_contacts_per_customer: int = Field(default=5, ge=1, description="Global contact cap across all channels")
    require_explicit_consent_for_reminder: bool = Field(default=True, description="Enforce opt-in consent for reminders")
    require_consent_for_payment_update: bool = Field(default=True, description="Enforce customer contactability for update requests")

    # 3. Escalation limits
    max_escalations_per_case: int = Field(default=1, ge=1, description="Max human escalation actions per case")
    min_days_overdue_for_escalation: int = Field(default=3, ge=0, description="Minimum aging before human escalation is eligible")
    min_amount_for_escalation: float = Field(default=500.0, ge=0.0, description="Minimum amount at risk for human escalation")

    # 4. Wait limits
    max_consecutive_wait_days: int = Field(default=14, ge=1, description="Maximum consecutive days in WAIT state before forced action")

    # 5. PCI Compliance boundary
    enforce_pci_tokenization_boundary: bool = Field(default=True, description="Ensure no direct handling of raw card PAN/CVV")

    # 6. Exploration & Safety limits
    enable_exploration_protection_for_vip: bool = Field(default=True, description="Restrict high-risk exploration on high-value accounts")
    vip_customer_value_threshold: float = Field(default=25000.0, ge=0.0, description="Customer value threshold for VIP protection")
