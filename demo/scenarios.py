"""Pre-configured deterministic demonstration scenarios."""
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from src.domain.actions import ActionType
from src.domain.case import (
    RecoveryCase,
    CustomerProfile,
    PaymentFailureCode,
    PaymentMethodType,
    CaseState,
)
from src.domain.events import PaymentFailureEvent
from src.verification.reconciliation import ReconciliationData


@dataclass
class DemoScenario:
    scenario_id: str
    name: str
    category: str
    description: str
    event: PaymentFailureEvent
    customer: CustomerProfile
    candidate_actions: Optional[List[ActionType]]
    expected_selected_action: ActionType
    reconciliation_data: Optional[ReconciliationData]
    simulate_duplicate: bool = False
    simulate_hallucination: bool = False


DEMO_SCENARIOS: Dict[str, DemoScenario] = {
    "scenario_a": DemoScenario(
        scenario_id="scenario_a",
        name="Scenario A: Standard Insufficient Funds (Auto-Retry & Settlement)",
        category="Happy Path Recovery",
        description="A loyal high-value subscriber encounters a soft decline. The Decision Engine computes LinUCB exploration bonus and two-step Q2, selects RETRY. Executor executes provisional retry, holding in IN_OBSERVATION. After window, settlement is ledger-confirmed to mark RESOLVED_RECOVERED.",
        event=PaymentFailureEvent(
            event_id="EVT-DEMO-01",
            customer_id="CUST-DEMO-01",
            invoice_id="CASE-DEMO-01",
            amount=1499.00,
            failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
            payment_method=PaymentMethodType.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
        ),
        customer=CustomerProfile(
            customer_id="CUST-DEMO-01",
            customer_value=9500.0,
            subscription_age_days=240,
            previous_success_rate=0.92,
            previous_contact_count=0,
            payment_method_type=PaymentMethodType.CREDIT_CARD,
            opt_in_email=True,
            opt_in_sms=True,
        ),
        candidate_actions=None,
        expected_selected_action=ActionType.RETRY,
        reconciliation_data=ReconciliationData(
            reconciliation_reference="REC-SETTLED-01",
            settlement_confirmed=True,
            gross_amount_settled=1499.00,
            net_amount_recovered=1499.00,
        ),
    ),
    "scenario_b": DemoScenario(
        scenario_id="scenario_b",
        name="Scenario B: Card Expired (Constrained Messaging & Update)",
        category="Payment Update Workflow",
        description="A customer credit card has expired. Policy Engine blocks immediate retry. Decision Engine selects PAYMENT_UPDATE. The LLM generates a constrained message which is deterministically verified by MessageValidator.",
        event=PaymentFailureEvent(
            event_id="EVT-DEMO-02",
            customer_id="CUST-DEMO-02",
            invoice_id="CASE-DEMO-02",
            amount=2400.00,
            failure_code=PaymentFailureCode.CARD_EXPIRED,
            payment_method=PaymentMethodType.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
        ),
        customer=CustomerProfile(
            customer_id="CUST-DEMO-02",
            customer_value=6000.0,
            subscription_age_days=365,
            previous_success_rate=0.88,
            previous_contact_count=0,
            payment_method_type=PaymentMethodType.CREDIT_CARD,
            opt_in_email=True,
            opt_in_sms=True,
        ),
        candidate_actions=None,
        expected_selected_action=ActionType.PAYMENT_UPDATE,
        reconciliation_data=None,
    ),
    "scenario_c": DemoScenario(
        scenario_id="scenario_c",
        name="Scenario C: Fraud Suspected (Hard Policy Decline & Escalation)",
        category="Risk & Policy Protection",
        description="A transaction is flagged with FRAUD_SUSPECTED. Hard decline policy rules prune RETRY and automated contacts. Decision Engine routes to ESCALATE with strictly zero LinUCB exploration bonus.",
        event=PaymentFailureEvent(
            event_id="EVT-DEMO-03",
            customer_id="CUST-DEMO-03",
            invoice_id="CASE-DEMO-03",
            amount=15000.00,
            failure_code=PaymentFailureCode.FRAUD_SUSPECTED,
            payment_method=PaymentMethodType.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
        ),
        customer=CustomerProfile(
            customer_id="CUST-DEMO-03",
            customer_value=15000.0,
            subscription_age_days=10,
            previous_success_rate=0.50,
            previous_contact_count=0,
            payment_method_type=PaymentMethodType.CREDIT_CARD,
            opt_in_email=True,
            opt_in_sms=False,
        ),
        candidate_actions=None,
        expected_selected_action=ActionType.ESCALATE,
        reconciliation_data=None,
    ),
    "scenario_d": DemoScenario(
        scenario_id="scenario_d",
        name="Scenario D: Contact Fatigue Capping (Dynamic Wait Cost)",
        category="Fatigue & Cadence Control",
        description="Customer has received multiple contacts recently. Communication frequency limits prevent further reminders. System evaluates dynamic wait cost C_wait = r_hold * days_waiting + r_delay * days_overdue and chooses WAIT.",
        event=PaymentFailureEvent(
            event_id="EVT-DEMO-04",
            customer_id="CUST-DEMO-04",
            invoice_id="CASE-DEMO-04",
            amount=850.00,
            failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
            payment_method=PaymentMethodType.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
        ),
        customer=CustomerProfile(
            customer_id="CUST-DEMO-04",
            customer_value=3000.0,
            subscription_age_days=90,
            previous_success_rate=0.70,
            previous_contact_count=5,
            payment_method_type=PaymentMethodType.CREDIT_CARD,
            opt_in_email=True,
            opt_in_sms=True,
        ),
        candidate_actions=[ActionType.WAIT, ActionType.REMINDER],
        expected_selected_action=ActionType.WAIT,
        reconciliation_data=None,
    ),
    "scenario_e": DemoScenario(
        scenario_id="scenario_e",
        name="Scenario E: Negative Sequence Value (Terminal STOP Action)",
        category="Cost & Recovery Optimization",
        description="When all active interventions yield negative expected two-step returns, STOP with Q(STOP)=0.0 deterministically wins, transitioning to RESOLVED_UNRECOVERABLE.",
        event=PaymentFailureEvent(
            event_id="EVT-DEMO-05",
            customer_id="CUST-DEMO-05",
            invoice_id="CASE-DEMO-05",
            amount=50.00,
            failure_code=PaymentFailureCode.GENERIC_DECLINE,
            payment_method=PaymentMethodType.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
        ),
        customer=CustomerProfile(
            customer_id="CUST-DEMO-05",
            customer_value=50.0,
            subscription_age_days=10,
            previous_success_rate=0.10,
            previous_contact_count=4,
            payment_method_type=PaymentMethodType.CREDIT_CARD,
            opt_in_email=False,
            opt_in_sms=False,
        ),
        candidate_actions=[ActionType.STOP],
        expected_selected_action=ActionType.STOP,
        reconciliation_data=None,
    ),
    "scenario_f": DemoScenario(
        scenario_id="scenario_f",
        name="Scenario F: Post-Settlement Chargeback (Reconciliation Invalidation)",
        category="Dispute Invalidation",
        description="A payment was initially executed, but downstream reconciliation detects a bank chargeback dispute. Final recovery is invalidated and terminal state is locked to RESOLVED_UNRECOVERABLE.",
        event=PaymentFailureEvent(
            event_id="EVT-DEMO-06",
            customer_id="CUST-DEMO-06",
            invoice_id="CASE-DEMO-06",
            amount=3200.00,
            failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
            payment_method=PaymentMethodType.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
        ),
        customer=CustomerProfile(
            customer_id="CUST-DEMO-06",
            customer_value=4000.0,
            subscription_age_days=150,
            previous_success_rate=0.80,
            previous_contact_count=0,
            payment_method_type=PaymentMethodType.CREDIT_CARD,
            opt_in_email=True,
            opt_in_sms=True,
        ),
        candidate_actions=None,
        expected_selected_action=ActionType.RETRY,
        reconciliation_data=ReconciliationData(
            reconciliation_reference="REC-CB-006",
            is_chargeback=True,
        ),
    ),
    "scenario_g": DemoScenario(
        scenario_id="scenario_g",
        name="Scenario G: Rogue LLM Hallucination Rejection (Validator Guardrail)",
        category="LLM Anti-Hallucination",
        description="Demonstrates the deterministic MessageValidator catching a rogue LLM candidate that hallucinated a wrong amount and wrong failure reason. Message is REJECTED while decision remains intact.",
        event=PaymentFailureEvent(
            event_id="EVT-DEMO-07",
            customer_id="CUST-DEMO-07",
            invoice_id="CASE-DEMO-07",
            amount=1200.00,
            failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
            payment_method=PaymentMethodType.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
        ),
        customer=CustomerProfile(
            customer_id="CUST-DEMO-07",
            customer_value=5000.0,
            subscription_age_days=100,
            previous_success_rate=0.80,
            previous_contact_count=0,
            payment_method_type=PaymentMethodType.CREDIT_CARD,
            opt_in_email=True,
            opt_in_sms=True,
        ),
        candidate_actions=[ActionType.PAYMENT_UPDATE],
        expected_selected_action=ActionType.PAYMENT_UPDATE,
        reconciliation_data=None,
        simulate_hallucination=True,
    ),
    "scenario_h": DemoScenario(
        scenario_id="scenario_h",
        name="Scenario H: Concurrent Duplicate Request (Atomic Idempotency Guard)",
        category="Idempotency Protection",
        description="Demonstrates atomic idempotency protection. The same (case_id, decision_id, action, attempt) key executed concurrently is safely deduplicated without duplicate financial execution.",
        event=PaymentFailureEvent(
            event_id="EVT-DEMO-08",
            customer_id="CUST-DEMO-08",
            invoice_id="CASE-DEMO-08",
            amount=1800.00,
            failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
            payment_method=PaymentMethodType.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
        ),
        customer=CustomerProfile(
            customer_id="CUST-DEMO-08",
            customer_value=7000.0,
            subscription_age_days=200,
            previous_success_rate=0.85,
            previous_contact_count=0,
            payment_method_type=PaymentMethodType.CREDIT_CARD,
            opt_in_email=True,
            opt_in_sms=True,
        ),
        candidate_actions=[ActionType.RETRY],
        expected_selected_action=ActionType.RETRY,
        reconciliation_data=None,
        simulate_duplicate=True,
    ),
}
