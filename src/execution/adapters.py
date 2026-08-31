"""Action execution adapters abstracting external systems and channel dispatch."""
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile
from src.execution.status import ExecutionStatus

DEFAULT_ADAPTER_VERSION: str = "adapter_v5.0.0"


class AdapterResponse(BaseModel):
    """Normalized response from an external action adapter."""
    status: ExecutionStatus
    reference_id: Optional[str] = None
    error_code: Optional[str] = None
    details: Dict[str, Any] = {}
    adapter_version: str = DEFAULT_ADAPTER_VERSION


class BaseActionAdapter(ABC):
    """Abstract base class for executing specific recovery actions."""

    @abstractmethod
    def execute(self, case: RecoveryCase, customer: CustomerProfile) -> AdapterResponse:
        """Dispatch action to external payment gateway, messaging provider, or workflow system."""
        pass


class PaymentRetryAdapter(BaseActionAdapter):
    """Dispatches automated charge retry against the payment gateway token."""

    def execute(self, case: RecoveryCase, customer: CustomerProfile) -> AdapterResponse:
        # Prototype deterministic gateway mock
        tx_id = f"tx_retry_{uuid.uuid4().hex[:8]}"
        return AdapterResponse(
            status=ExecutionStatus.SUCCESS,
            reference_id=tx_id,
            details={"gateway": "mock_pci_gateway", "amount_charged": float(case.amount_at_risk)},
        )


class PaymentUpdateAdapter(BaseActionAdapter):
    """Generates secure self-serve hosted portal link for payment method updates."""

    def execute(self, case: RecoveryCase, customer: CustomerProfile) -> AdapterResponse:
        link_id = f"portal_link_{uuid.uuid4().hex[:8]}"
        return AdapterResponse(
            status=ExecutionStatus.SUCCESS,
            reference_id=link_id,
            details={"portal_url": f"https://pay.recover.io/update/{link_id}", "customer_id": customer.customer_id},
        )


class ReminderAdapter(BaseActionAdapter):
    """Dispatches outbound communication reminder via email or SMS."""

    def execute(self, case: RecoveryCase, customer: CustomerProfile) -> AdapterResponse:
        msg_id = f"msg_reminder_{uuid.uuid4().hex[:8]}"
        channel = "email" if customer.opt_in_email else "sms"
        return AdapterResponse(
            status=ExecutionStatus.SUCCESS,
            reference_id=msg_id,
            details={"channel": channel, "recipient": customer.customer_id},
        )


class WaitAdapter(BaseActionAdapter):
    """Schedules passive holding interval without immediate payment or contact attempts."""

    def execute(self, case: RecoveryCase, customer: CustomerProfile) -> AdapterResponse:
        sched_id = f"wait_hold_{case.case_id}_d{case.days_waiting + 1}"
        return AdapterResponse(
            status=ExecutionStatus.SUCCESS,
            reference_id=sched_id,
            details={"hold_duration_days": 1, "next_evaluation_day": case.days_overdue + 1},
        )


class EscalateAdapter(BaseActionAdapter):
    """Creates high-touch operational review ticket for manual agent intervention."""

    def execute(self, case: RecoveryCase, customer: CustomerProfile) -> AdapterResponse:
        ticket_id = f"TICKET-OPS-{uuid.uuid4().hex[:6].upper()}"
        return AdapterResponse(
            status=ExecutionStatus.SUCCESS,
            reference_id=ticket_id,
            details={"assigned_queue": "high_value_ops", "priority": "P1" if customer.customer_value > 5000 else "P2"},
        )


class StopAdapter(BaseActionAdapter):
    """Terminal action: records graceful cessation of recovery efforts (no-op)."""

    def execute(self, case: RecoveryCase, customer: CustomerProfile) -> AdapterResponse:
        term_id = f"term_{case.case_id}"
        return AdapterResponse(
            status=ExecutionStatus.SUCCESS,
            reference_id=term_id,
            details={"reason": "TERMINAL_STOP", "recovery_ended": True},
        )
