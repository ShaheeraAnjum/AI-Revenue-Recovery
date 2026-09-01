"""End-to-end revenue recovery orchestration pipeline connecting frozen domain components."""
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile, CaseState
from src.domain.events import PaymentFailureEvent
from src.engine.decision import DecisionEngine, DecisionResult
from src.execution.executor import ActionExecutor, ExecutionResult
from src.verification.verifier import VerificationEngine
from src.verification.observation import ObservationRecord
from src.verification.reconciliation import ReconciliationData
from src.messaging.service import LLMMessageService
from src.messaging.templates import APPROVED_TEMPLATES
from src.orchestration.schema import OrchestrationCycleResult


class RevenueRecoveryOrchestrator:
    """Deterministic coordinator executing the complete revenue recovery lifecycle:
    PaymentFailureEvent -> RecoveryCase -> DecisionEngine -> Executor -> Observation -> Reconciliation -> Constrained Messaging.
    """

    def __init__(
        self,
        decision_engine: Optional[DecisionEngine] = None,
        executor: Optional[ActionExecutor] = None,
        verification_engine: Optional[VerificationEngine] = None,
        messaging_service: Optional[LLMMessageService] = None,
    ):
        self.decision_engine = decision_engine or DecisionEngine()
        self.executor = executor or ActionExecutor()
        self.verification_engine = verification_engine or VerificationEngine()
        self.messaging_service = messaging_service or LLMMessageService()

    def ingest_failure_event(
        self,
        event: PaymentFailureEvent,
        customer: CustomerProfile,
    ) -> RecoveryCase:
        """Ingest a payment failure event to create or activate a recovery case."""
        case = RecoveryCase(
            case_id=event.invoice_id or f"CASE-{uuid.uuid4().hex[:8].upper()}",
            customer_id=customer.customer_id,
            amount_at_risk=event.amount,
            failure_code=event.failure_code,
            days_overdue=0,
            days_waiting=0,
            state=CaseState.ACTIVE,
        )
        return case

    def process_recovery_cycle(
        self,
        case: RecoveryCase,
        customer: CustomerProfile,
        candidate_actions: Optional[List[ActionType]] = None,
        random_seed: Optional[int] = None,
        decision_id: Optional[str] = None,
        preferred_template_id: Optional[str] = None,
    ) -> OrchestrationCycleResult:
        """Coordinate one deterministic recovery cycle without inventing decision logic."""
        cycle_id = f"CYC-{uuid.uuid4().hex[:8].upper()}"

        # 1. Decision Engine evaluates policy and selects action via single argmax
        dec_res: DecisionResult = self.decision_engine.decide(
            case=case,
            customer=customer,
            candidate_actions=candidate_actions,
            decision_id=decision_id,
            random_seed=random_seed,
        )

        # Handle empty candidate / no-action result
        if dec_res.selected_action is None:
            return OrchestrationCycleResult(
                cycle_id=cycle_id,
                case_id=case.case_id,
                decision_id=dec_res.decision_id,
                selected_action=None,
                decision_result=dec_res,
                case_state=case.state,
            )

        selected_act = dec_res.selected_action
        attempt = case.retry_attempt_count + case.reminder_count + case.escalation_count + 1

        # 2. Safely execute selected action via idempotent executor
        exec_res: ExecutionResult = self.executor.execute(
            action=selected_act,
            case=case,
            customer=customer,
            decision_id=dec_res.decision_id,
            attempt=attempt,
        )

        # Update case attempt counters based on executed action
        if selected_act == ActionType.RETRY:
            case.retry_attempt_count += 1
        elif selected_act == ActionType.REMINDER:
            case.reminder_count += 1
        elif selected_act == ActionType.ESCALATE:
            case.escalation_count += 1
        elif selected_act == ActionType.WAIT:
            case.days_waiting += 1
        case.last_action = selected_act

        # 3. Create provisional observation in verification engine (case -> IN_OBSERVATION)
        obs_rec: ObservationRecord = self.verification_engine.create_observation(
            execution_result=exec_res,
            case=case,
        )

        # 4. Optional constrained customer messaging (if action has an applicable template)
        delivered_msg = None
        val_res = None
        msg_audit = None

        template_id = preferred_template_id
        if template_id is None:
            # Auto-select approved template for action
            if selected_act == ActionType.PAYMENT_UPDATE:
                template_id = "TMPL_PAYMENT_UPDATE_EMAIL_V1"
            elif selected_act == ActionType.REMINDER:
                template_id = "TMPL_REMINDER_EMAIL_V1"
            elif selected_act == ActionType.RETRY:
                template_id = "TMPL_RETRY_NOTICE_EMAIL_V1"

        if template_id:
            delivered_msg, val_res, msg_audit = self.messaging_service.generate_and_validate(
                case=case,
                customer=customer,
                selected_action=selected_act,
                template_id=template_id,
                decision_id=dec_res.decision_id,
            )

        return OrchestrationCycleResult(
            cycle_id=cycle_id,
            case_id=case.case_id,
            decision_id=dec_res.decision_id,
            idempotency_key=exec_res.idempotency_key,
            selected_action=selected_act,
            decision_result=dec_res,
            execution_result=exec_res,
            observation_record=obs_rec,
            delivered_message=delivered_msg,
            message_validation=val_res,
            message_audit=msg_audit,
            case_state=case.state,
        )

    def reconcile_case(
        self,
        idempotency_key: str,
        case: RecoveryCase,
        reconciliation_data: ReconciliationData,
        as_of_time: Optional[datetime] = None,
    ) -> ObservationRecord:
        """Coordinate post-observation financial reconciliation."""
        return self.verification_engine.reconcile(
            idempotency_key=idempotency_key,
            case=case,
            reconciliation_data=reconciliation_data,
            as_of_time=as_of_time,
        )
