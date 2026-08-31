"""Recovery case and customer entities."""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, NamedTuple
from pydantic import BaseModel, Field
from src.domain.actions import ActionType


class CaseState(str, Enum):
    """Lifecycle states for a recovery case."""
    DETECTED = "DETECTED"
    ACTIVE = "ACTIVE"
    IN_OBSERVATION = "IN_OBSERVATION"
    RESOLVED_RECOVERED = "RESOLVED_RECOVERED"
    RESOLVED_UNRECOVERABLE = "RESOLVED_UNRECOVERABLE"


class PaymentFailureCode(str, Enum):
    """Standardized gateway failure reason codes."""
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    CARD_EXPIRED = "CARD_EXPIRED"
    DO_NOT_HONOR = "DO_NOT_HONOR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"
    INVALID_CARD_NUMBER = "INVALID_CARD_NUMBER"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    GENERIC_DECLINE = "GENERIC_DECLINE"


class PaymentMethodType(str, Enum):
    """Payment method types for customer context."""
    CREDIT_CARD = "CREDIT_CARD"
    DEBIT_CARD = "DEBIT_CARD"
    ACH_DIRECT_DEBIT = "ACH_DIRECT_DEBIT"
    DIGITAL_WALLET = "DIGITAL_WALLET"
    UPI = "UPI"


class CustomerProfile(BaseModel):
    """Customer attributes and compliance consent."""
    customer_id: str
    customer_value: float = Field(..., ge=0.0, description="Historical customer lifetime value or ARR contribution")
    subscription_age_days: int = Field(..., ge=0, description="Age of subscription in days")
    previous_success_rate: float = Field(..., ge=0.0, le=1.0, description="Historical recovery/payment success rate [0, 1]")
    previous_contact_count: int = Field(default=0, ge=0, description="Number of outbound communications received")
    payment_method_type: PaymentMethodType = PaymentMethodType.CREDIT_CARD
    opt_in_email: bool = True
    opt_in_sms: bool = True
    active_recovery_cases: int = 0
    audit_segment: str = Field(default="standard", description="Approved segment for fairness monitoring")


class RecoveryCase(BaseModel):
    """Individual recovery case tracking."""
    case_id: str
    customer_id: str
    amount_at_risk: float = Field(..., gt=0.0, description="Outstanding amount needing recovery")
    failure_code: PaymentFailureCode
    days_overdue: int = Field(default=0, ge=0)
    state: CaseState = CaseState.DETECTED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retry_attempt_count: int = Field(default=0, ge=0)
    reminder_count: int = Field(default=0, ge=0)
    last_action: Optional[ActionType] = None
    days_waiting: int = Field(default=0, ge=0, description="Number of consecutive days in WAIT state")


class IdempotencyKey(NamedTuple):
    """Exact 4-tuple key required for idempotent action execution: (case_id, decision_id, action, attempt)."""
    case_id: str
    decision_id: str
    action: ActionType
    attempt: int

    def to_string(self) -> str:
        return f"{self.case_id}:{self.decision_id}:{self.action.value}:{self.attempt}"
