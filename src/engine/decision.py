"""Final current-action decision engine performing policy-filtered single argmax action selection."""
import uuid
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.domain.actions import ActionType, is_supported_action
from src.domain.case import RecoveryCase, CustomerProfile, IdempotencyKey
from src.context.schema import ContextFeatures
from src.context.builder import ContextBuilder
from src.policy.engine import PolicyEngine, PolicyDecision
from src.actions.generator import CandidateActionGenerator
from src.models.linucb import LinUCBValueModel
from src.engine.two_step import TwoStepValueEngine, TwoStepScoringResult
from src.audit.schema import (
    DecisionAuditRecord,
    ActionScoringDetail,
    EstimationMethod,
    VersionConfig,
)

# Canonical tie-breaking priority order
CANONICAL_TIE_BREAKER_PRIORITY: List[ActionType] = [
    ActionType.RETRY,
    ActionType.PAYMENT_UPDATE,
    ActionType.REMINDER,
    ActionType.WAIT,
    ActionType.ESCALATE,
    ActionType.STOP,
]


class DecisionEngineConfig(BaseModel):
    """Configuration hyperparameters and versioning for final action decisioning."""
    alpha: float = Field(default=1.0, ge=0.0, description="Exploration bonus scaling factor for current action layer")
    tie_breaker_priority: List[ActionType] = Field(default_factory=lambda: list(CANONICAL_TIE_BREAKER_PRIORITY))
    versions: VersionConfig = Field(default_factory=VersionConfig)


class DecisionResult(BaseModel):
    """Complete output of the Decision Engine containing selected action, idempotency key, and audit log."""
    decision_id: str
    case_id: str
    customer_id: str
    selected_action: ActionType
    idempotency_key: str
    q2_values: Dict[ActionType, float]
    exploration_bonuses: Dict[ActionType, float]
    final_q_values: Dict[ActionType, float]
    estimation_methods: Dict[ActionType, EstimationMethod]
    allowed_actions: List[ActionType]
    prohibited_actions: Dict[ActionType, str]
    audit_record: DecisionAuditRecord


class DecisionEngine:
    """Decision Engine executing the frozen current-action selection pipeline:
    1. Candidate Action Generation
    2. Policy / Safety Filtering (Pre-scoring)
    3. Two-Step Sequence Value Evaluation Q2(s, a) for A_allowed only
    4. Current-Layer LinUCB Exploration Bonus Calculation B(x, a)
    5. Single Argmax Selection: a* = argmax_{a in A_allowed} [ Q2(s, a) + B(x, a) ]
    6. Full Audit Trail Recording with 8 Version Dimensions
    """

    def __init__(
        self,
        config: Optional[DecisionEngineConfig] = None,
        candidate_generator: Optional[CandidateActionGenerator] = None,
        policy_engine: Optional[PolicyEngine] = None,
        two_step_engine: Optional[TwoStepValueEngine] = None,
        linucb_model: Optional[LinUCBValueModel] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.config = config or DecisionEngineConfig()
        self.generator = candidate_generator or CandidateActionGenerator()
        self.policy_engine = policy_engine or PolicyEngine()
        self.two_step_engine = two_step_engine or TwoStepValueEngine()
        self.linucb = linucb_model or self.two_step_engine.linucb
        self.context_builder = context_builder or ContextBuilder()
        self.alpha = self.config.alpha
        self.versions = self.config.versions

    def decide(
        self,
        case: RecoveryCase,
        customer: CustomerProfile,
        candidate_actions: Optional[List[ActionType]] = None,
        decision_id: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> DecisionResult:
        """Select optimal compliance-approved action via argmax[Q2 + B]."""
        d_id = decision_id or f"DEC-{uuid.uuid4().hex[:12].upper()}"

        # 1. Candidate action generation
        candidates = (
            list(candidate_actions)
            if candidate_actions is not None
            else self.generator.generate_candidates(case, customer)
        )

        # 2. Pre-scoring Policy / Safety Engine filtering
        policy_decision: PolicyDecision = self.policy_engine.evaluate(
            case=case,
            customer=customer,
            candidate_actions=candidates,
        )
        allowed = policy_decision.allowed_actions
        prohibited = policy_decision.prohibited_actions

        # If no actions allowed, fallback to STOP as terminal safeguard
        if not allowed:
            allowed = [ActionType.STOP]

        # 3. Context extraction
        context = self.context_builder.build_context(case, customer)

        # 4. Evaluate Q2 sequence values strictly for allowed actions
        q2_results: Dict[ActionType, TwoStepScoringResult] = self.two_step_engine.evaluate_allowed_actions(
            allowed_actions=allowed,
            case=case,
            customer=customer,
        )

        # 5. Compute current-layer exploration bonus B(x, a) and final scores
        action_details: Dict[ActionType, ActionScoringDetail] = {}
        q1_values: Dict[ActionType, float] = {}
        q2_values: Dict[ActionType, float] = {}
        exploration_bonuses: Dict[ActionType, float] = {}
        final_q_values: Dict[ActionType, float] = {}
        estimation_methods: Dict[ActionType, EstimationMethod] = {}
        confidences: Dict[ActionType, float] = {}

        for a in allowed:
            scoring_res = q2_results[a]
            q2_val = scoring_res.q2_sequence_value
            q1_val = scoring_res.q1_base_value
            est_method = scoring_res.estimation_method

            # Exploration bonus B(x, a) = alpha * sqrt(x^T A^-1 x) ONLY for supported actions
            if is_supported_action(a):
                bonus = self.linucb.compute_exploration_bonus(
                    context, a, custom_alpha=self.alpha
                )
                conf = 0.90
            else:
                # ESCALATE and STOP have strictly ZERO LinUCB exploration bonus
                bonus = 0.0
                conf = 0.65 if a == ActionType.ESCALATE else 1.0

            # STOP has Q2=0.0 and B=0.0 -> score = 0.0
            if a == ActionType.STOP:
                q2_val = 0.0
                bonus = 0.0
                final_score = 0.0
            else:
                final_score = float(q2_val + bonus)

            q1_values[a] = q1_val
            q2_values[a] = q2_val
            exploration_bonuses[a] = bonus
            final_q_values[a] = final_score
            estimation_methods[a] = est_method
            confidences[a] = conf

            action_details[a] = ActionScoringDetail(
                action=a,
                is_allowed=True,
                rejection_reason=None,
                estimation_method=est_method,
                q1_base_value=q1_val,
                q2_sequence_value=q2_val,
                exploration_bonus=bonus,
                final_q_value=final_score,
                confidence=conf,
            )

        # Also document prohibited actions in action_details
        for a_proh, reason in prohibited.items():
            action_details[a_proh] = ActionScoringDetail(
                action=a_proh,
                is_allowed=False,
                rejection_reason=reason,
            )

        # 6. Final Argmax Selection over allowed actions with deterministic tie-breaking
        best_score = -float("inf")
        selected_action = ActionType.STOP

        # Sort allowed actions by tie-breaker priority
        priority_map = {act: idx for idx, act in enumerate(self.config.tie_breaker_priority)}
        sorted_allowed = sorted(allowed, key=lambda act: priority_map.get(act, 999))

        for a in sorted_allowed:
            score = final_q_values[a]
            if score > best_score:
                best_score = score
                selected_action = a

        # 7. Form exact idempotency key
        attempt = case.retry_attempt_count + case.reminder_count + case.escalation_count + 1
        idempotency_key = IdempotencyKey(
            case_id=case.case_id,
            decision_id=d_id,
            action=selected_action,
            attempt=attempt,
        )

        # 8. Construct Audit Record with all 8 version dimensions
        audit_record = DecisionAuditRecord(
            decision_id=d_id,
            case_id=case.case_id,
            customer_id=customer.customer_id,
            context_features=context.to_dict(),
            feature_vector=list(context.feature_vector),
            candidate_actions=candidates,
            allowed_actions=allowed,
            prohibited_actions=prohibited,
            action_details=action_details,
            q1_values=q1_values,
            q2_values=q2_values,
            exploration_bonuses=exploration_bonuses,
            final_q_values=final_q_values,
            estimation_methods=estimation_methods,
            confidences=confidences,
            selected_action=selected_action,
            random_seed=random_seed,
            policy_version=self.policy_engine.config.policy_version,
            value_model_version=self.linucb.version,
            transition_model_version=self.two_step_engine.transition_model.version,
            propensity_model_version=self.versions.propensity_model_version,
            fairness_policy_version=self.versions.fairness_policy_version,
            message_policy_version=self.versions.message_policy_version,
            feature_schema_version=context.feature_schema_version,
            exploration_config_version=self.versions.exploration_config_version,
        )

        return DecisionResult(
            decision_id=d_id,
            case_id=case.case_id,
            customer_id=customer.customer_id,
            selected_action=selected_action,
            idempotency_key=idempotency_key.to_string(),
            q2_values=q2_values,
            exploration_bonuses=exploration_bonuses,
            final_q_values=final_q_values,
            estimation_methods=estimation_methods,
            allowed_actions=allowed,
            prohibited_actions=prohibited,
            audit_record=audit_record,
        )
