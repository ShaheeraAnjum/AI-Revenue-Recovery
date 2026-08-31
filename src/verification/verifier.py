"""Deterministic verification and reconciliation engine enforcing observation windows."""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CaseState
from src.domain.verification import VerificationStatus
from src.execution.status import ExecutionStatus
from src.execution.executor import ExecutionResult
from src.verification.observation import (
    ObservationRecord,
    ObservationConfig,
    ReconciliationStatus,
    DEFAULT_OBSERVATION_VERSION,
)
from src.verification.reconciliation import ReconciliationData, DEFAULT_RECONCILIATION_VERSION


class VerificationEngine:
    """Processes provisional execution results through observation windows and reconciliation."""

    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()
        self.observations: Dict[str, ObservationRecord] = {}

    def create_observation(
        self,
        execution_result: ExecutionResult,
        case: RecoveryCase,
        start_time: Optional[datetime] = None,
        custom_window_seconds: Optional[int] = None,
    ) -> ObservationRecord:
        """Record initial provisional observation from execution result.
        CRITICAL INVARIANT: Never sets final_outcome to RESOLVED_RECOVERED.
        """
        obs_start = start_time or datetime.now(timezone.utc)
        window_sec = (
            custom_window_seconds
            if custom_window_seconds is not None
            else self.config.default_window_seconds
        )
        deadline = obs_start + timedelta(seconds=window_sec)
        obs_id = f"OBS-{uuid.uuid4().hex[:10].upper()}"

        # Determine initial provisional verification status
        if execution_result.execution_status == ExecutionStatus.SUCCESS:
            v_status = VerificationStatus.UNKNOWN  # Pending observation/settlement
            case.state = CaseState.IN_OBSERVATION
            final_out = None
        elif execution_result.execution_status == ExecutionStatus.DECLINE:
            v_status = VerificationStatus.DECLINE
            case.state = CaseState.ACTIVE
            final_out = None
        elif execution_result.execution_status == ExecutionStatus.TIMEOUT:
            v_status = VerificationStatus.TIMEOUT
            final_out = None
        else:
            v_status = VerificationStatus.UNKNOWN
            final_out = None

        if execution_result.action == ActionType.STOP:
            case.state = CaseState.RESOLVED_UNRECOVERABLE
            final_out = CaseState.RESOLVED_UNRECOVERABLE

        record = ObservationRecord(
            observation_id=obs_id,
            case_id=execution_result.case_id,
            decision_id=execution_result.decision_id,
            action=execution_result.action,
            attempt=execution_result.attempt,
            idempotency_key=execution_result.idempotency_key,
            execution_status=execution_result.execution_status,
            execution_timestamp=execution_result.execution_timestamp,
            observation_start=obs_start,
            observation_window_seconds=window_sec,
            observation_deadline=deadline,
            is_provisional=True,
            verification_status=v_status,
            reconciliation_status=ReconciliationStatus.PENDING,
            final_outcome=final_out,
            observation_version=self.config.observation_version,
            metadata={"adapter_reference": execution_result.reference_id},
        )

        self.observations[record.idempotency_key] = record
        return record

    def reconcile(
        self,
        idempotency_key: str,
        case: RecoveryCase,
        reconciliation_data: ReconciliationData,
        as_of_time: Optional[datetime] = None,
    ) -> ObservationRecord:
        """Reconcile observation after window elapses using settlement/refund/chargeback ledger."""
        if idempotency_key not in self.observations:
            raise ValueError(f"Observation for idempotency key '{idempotency_key}' not found")

        record = self.observations[idempotency_key]
        now = as_of_time or datetime.now(timezone.utc)

        # 1. Enforce observation window
        if not record.is_window_elapsed(now):
            # Window not elapsed -> Cannot finalize recovery!
            record.verification_status = VerificationStatus.UNKNOWN
            record.reconciliation_status = ReconciliationStatus.PENDING
            record.final_outcome = None
            return record

        # 2. Process reconciliation outcomes when execution was SUCCESS
        if record.execution_status in {ExecutionStatus.SUCCESS, ExecutionStatus.DUPLICATE}:
            if reconciliation_data.is_chargeback:
                record.verification_status = VerificationStatus.RECONCILIATION
                record.reconciliation_status = ReconciliationStatus.CHARGEBACK
                record.final_outcome = CaseState.RESOLVED_UNRECOVERABLE
                case.state = CaseState.RESOLVED_UNRECOVERABLE
                record.is_provisional = False

            elif reconciliation_data.is_refunded:
                record.verification_status = VerificationStatus.RECONCILIATION
                record.reconciliation_status = ReconciliationStatus.REFUNDED
                record.final_outcome = CaseState.RESOLVED_UNRECOVERABLE
                case.state = CaseState.RESOLVED_UNRECOVERABLE
                record.is_provisional = False

            elif reconciliation_data.settlement_confirmed:
                record.verification_status = VerificationStatus.SUCCESS
                record.reconciliation_status = ReconciliationStatus.SETTLED
                record.final_outcome = CaseState.RESOLVED_RECOVERED
                case.state = CaseState.RESOLVED_RECOVERED
                record.is_provisional = False

            else:
                # Incomplete or unconfirmed settlement
                record.verification_status = VerificationStatus.UNKNOWN
                record.reconciliation_status = ReconciliationStatus.FAILED
                record.final_outcome = None
                record.is_provisional = True

        elif record.execution_status == ExecutionStatus.DECLINE:
            record.verification_status = VerificationStatus.DECLINE
            record.reconciliation_status = ReconciliationStatus.FAILED
            record.final_outcome = CaseState.ACTIVE
            case.state = CaseState.ACTIVE
            record.is_provisional = False

        else:
            record.verification_status = VerificationStatus.UNKNOWN
            record.reconciliation_status = ReconciliationStatus.PENDING
            record.final_outcome = None
            record.is_provisional = True

        record.metadata["reconciliation_reference"] = reconciliation_data.reconciliation_reference
        record.metadata["net_recovered"] = reconciliation_data.net_amount_recovered
        return record
