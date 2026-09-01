"""Approved message template registry with strict factual requirements."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import PaymentFailureCode
from src.messaging.schema import CommunicationChannel

DEFAULT_TEMPLATE_VERSION: str = "tmpl_v5.0.0"


class MessageTemplate(BaseModel):
    """Approved message template definition specifying compliance and factual boundaries."""
    template_id: str
    template_version: str = DEFAULT_TEMPLATE_VERSION
    applicable_actions: List[ActionType]
    channel: CommunicationChannel = CommunicationChannel.EMAIL
    requires_consent: bool = True
    required_facts: List[str] = Field(
        default_factory=lambda: ["case_id", "amount", "failure_reason"],
        description="Factual fields that must be present in the body"
    )
    allowed_failure_codes: List[PaymentFailureCode] = Field(
        default_factory=lambda: list(PaymentFailureCode)
    )
    template_format: str


# Canonical Approved Template Registry
APPROVED_TEMPLATES: Dict[str, MessageTemplate] = {
    "TMPL_PAYMENT_UPDATE_EMAIL_V1": MessageTemplate(
        template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
        applicable_actions=[ActionType.PAYMENT_UPDATE],
        channel=CommunicationChannel.EMAIL,
        requires_consent=True,
        required_facts=["case_id", "amount", "failure_reason"],
        template_format="Dear customer, your payment of {amount} for case {case_id} could not be processed due to {failure_reason}. Please update your payment method.",
    ),
    "TMPL_REMINDER_EMAIL_V1": MessageTemplate(
        template_id="TMPL_REMINDER_EMAIL_V1",
        applicable_actions=[ActionType.REMINDER],
        channel=CommunicationChannel.EMAIL,
        requires_consent=True,
        required_facts=["case_id", "amount", "failure_reason"],
        template_format="Reminder: Payment of {amount} for reference {case_id} is overdue due to {failure_reason}. Please review your billing details.",
    ),
    "TMPL_REMINDER_SMS_V1": MessageTemplate(
        template_id="TMPL_REMINDER_SMS_V1",
        applicable_actions=[ActionType.REMINDER],
        channel=CommunicationChannel.SMS,
        requires_consent=True,
        required_facts=["case_id", "amount"],
        template_format="Reminder: Payment of {amount} for {case_id} is due. Please check your account.",
    ),
    "TMPL_RETRY_NOTICE_EMAIL_V1": MessageTemplate(
        template_id="TMPL_RETRY_NOTICE_EMAIL_V1",
        applicable_actions=[ActionType.RETRY],
        channel=CommunicationChannel.EMAIL,
        requires_consent=True,
        required_facts=["case_id", "amount"],
        template_format="Notice: We are re-attempting your scheduled payment of {amount} for case {case_id}.",
    ),
}
