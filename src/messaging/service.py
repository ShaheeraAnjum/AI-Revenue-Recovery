"""Constrained messaging service generating, validating, and auditing customer communications."""
import uuid
from typing import Optional
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile
from src.messaging.schema import (
    CandidateMessage,
    ValidationResult,
    MessageAuditRecord,
    MessageValidationStatus,
    DEFAULT_MESSAGE_POLICY_VERSION,
)
from src.messaging.templates import MessageTemplate, APPROVED_TEMPLATES
from src.messaging.validator import MessageValidator
from src.messaging.llm_client import BaseLLMClient, MockLLMClient


class LLMMessageService:
    """Orchestrates constrained LLM message generation and deterministic validation."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        validator: Optional[MessageValidator] = None,
    ):
        self.llm_client = llm_client or MockLLMClient()
        self.validator = validator or MessageValidator()

    def generate_and_validate(
        self,
        case: RecoveryCase,
        customer: CustomerProfile,
        selected_action: ActionType,
        template_id: str,
        decision_id: str,
        override_candidate_message: Optional[CandidateMessage] = None,
    ) -> tuple[Optional[CandidateMessage], ValidationResult, MessageAuditRecord]:
        """Generate message from LLM and validate strictly against ground truth.
        IMPORTANT: This service CANNOT modify selected_action or case financials.
        """
        template = APPROVED_TEMPLATES.get(template_id)
        if template is None:
            val_res = ValidationResult(
                is_approved=False,
                status=MessageValidationStatus.REJECTED,
                rejection_reasons=["UNAPPROVED_TEMPLATE"],
            )
            audit = MessageAuditRecord(
                message_id=f"MSG-{uuid.uuid4().hex[:8].upper()}",
                case_id=case.case_id,
                decision_id=decision_id,
                selected_action=selected_action,
                template_id=template_id,
                template_version="unknown",
                generated_message="",
                validation_status=MessageValidationStatus.REJECTED,
                rejection_reasons=["UNAPPROVED_TEMPLATE"],
            )
            return None, val_res, audit

        # 1. Generate candidate message from LLM (or use candidate payload)
        if override_candidate_message is not None:
            candidate = override_candidate_message
        else:
            candidate = self.llm_client.generate_message(
                template=template,
                case=case,
                customer=customer,
                selected_action=selected_action,
            )

        # 2. Run deterministic validator
        val_res = self.validator.validate(
            message=candidate,
            case=case,
            customer=customer,
            selected_action=selected_action,
            template=template,
        )

        # 3. Produce audit record
        audit = MessageAuditRecord(
            message_id=f"MSG-{uuid.uuid4().hex[:8].upper()}",
            case_id=case.case_id,
            decision_id=decision_id,
            selected_action=selected_action,
            template_id=template.template_id,
            template_version=template.template_version,
            generated_message=candidate.body_text,
            validation_status=val_res.status,
            rejection_reasons=[r.value if hasattr(r, "value") else str(r) for r in val_res.rejection_reasons],
            validator_version=val_res.validator_version,
        )

        delivered_message = candidate if val_res.is_approved else None
        return delivered_message, val_res, audit
