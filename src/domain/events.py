"""Billing and payment events triggering recovery pipelines."""
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from src.domain.case import PaymentFailureCode, PaymentMethodType


class BillingEvent(BaseModel):
    """General billing lifecycle event."""
    event_id: str
    customer_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    amount: float
    currency: str = "USD"


class PaymentFailureEvent(BaseModel):
    """Payment failure event emitted by billing gateway."""
    event_id: str
    customer_id: str
    invoice_id: str
    amount: float = Field(..., gt=0.0)
    failure_code: PaymentFailureCode
    payment_method: PaymentMethodType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_gateway_message: str = ""
