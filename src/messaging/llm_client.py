"""LLM client interface and deterministic mock client for offline verification."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile
from src.messaging.schema import CandidateMessage
from src.messaging.templates import MessageTemplate


class BaseLLMClient(ABC):
    """Abstract LLM provider client interface."""

    @abstractmethod
    def generate_message(
        self,
        template: MessageTemplate,
        case: RecoveryCase,
        customer: CustomerProfile,
        selected_action: ActionType,
    ) -> CandidateMessage:
        """Generate message strictly bounded by template and factual inputs."""
        pass


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM client producing factual, template-compliant messages."""

    def generate_message(
        self,
        template: MessageTemplate,
        case: RecoveryCase,
        customer: CustomerProfile,
        selected_action: ActionType,
    ) -> CandidateMessage:
        body = template.template_format.format(
            amount=f"{case.amount_at_risk:.2f}",
            case_id=case.case_id,
            failure_reason=case.failure_code.value.replace("_", " ").lower(),
            portal_url=f"https://pay.recover.io/update/{case.case_id}",
        )
        return CandidateMessage(
            template_id=template.template_id,
            body_text=body,
            stated_amount=float(case.amount_at_risk),
            stated_failure_reason=case.failure_code.value,
            stated_action=selected_action,
        )
