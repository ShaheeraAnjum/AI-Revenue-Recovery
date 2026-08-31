"""Two-step sequence value engine implementing Q2(s, a) = R + gamma * sum(P * V1) - C."""
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.domain.actions import ActionType, is_supported_action, is_restricted_action
from src.domain.case import RecoveryCase, CustomerProfile, CaseState
from src.context.schema import ContextFeatures
from src.context.builder import ContextBuilder
from src.models.linucb import LinUCBValueModel
from src.models.transition import (
    ActionConditionalTransitionModel,
    RecoveryNextState,
    TransitionDistribution,
)
from src.engine.costs import ActionCostCalculator, CostConfig
from src.engine.heuristic import EscalateHeuristicModel, EscalateHeuristicConfig
from src.audit.schema import EstimationMethod

DEFAULT_ENGINE_CONFIG_VERSION: str = "engine_v5.0.0"


class TwoStepEngineConfig(BaseModel):
    """Configuration for two-step sequence value calculation."""
    engine_version: str = DEFAULT_ENGINE_CONFIG_VERSION
    gamma: float = Field(default=0.95, ge=0.0, le=1.0, description="Discount factor gamma in [0, 1]")


class TwoStepScoringResult(BaseModel):
    """Audit-ready intermediate scoring breakdown for a single evaluated action."""
    action: ActionType
    estimation_method: EstimationMethod
    q1_base_value: float = Field(..., description="Base value Q1(s, a) for current state")
    immediate_reward: float = Field(..., description="Action-dependent immediate reward R(s, a)")
    action_cost: float = Field(..., description="Action-specific cost C(s, a)")
    transition_probabilities: Dict[RecoveryNextState, float] = Field(
        ..., description="Transition probabilities P(s' | s, a)"
    )
    future_v1_values: Dict[RecoveryNextState, float] = Field(
        ..., description="Lookahead base policy values V1(s') = max Q1(s', a') (strictly NO exploration bonus)"
    )
    expected_future_value: float = Field(..., description="Expectation sum_s' P(s' | s, a) * V1(s')")
    discounted_future_value: float = Field(..., description="gamma * sum_s' P(s' | s, a) * V1(s')")
    q2_sequence_value: float = Field(..., description="Full sequence value Q2(s, a) = R + gamma * E[V1] - C")


class TwoStepValueEngine:
    """Calculates Two-Step Sequence Value Q2(s, a) across compliance-approved allowed actions.
    Enforces strict mathematical separation:
    - Base Q1 is used for future lookahead states V1(s') = max_a' Q1(s', a') (zero future exploration).
    - Current exploration bonus B(x, a) is NOT computed here (deferred to decision layer).
    """

    def __init__(
        self,
        config: Optional[TwoStepEngineConfig] = None,
        linucb_model: Optional[LinUCBValueModel] = None,
        transition_model: Optional[ActionConditionalTransitionModel] = None,
        cost_calculator: Optional[ActionCostCalculator] = None,
        heuristic_model: Optional[EscalateHeuristicModel] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.config = config or TwoStepEngineConfig()
        if not (0.0 <= self.config.gamma <= 1.0):
            raise ValueError(f"Discount factor gamma must be in [0, 1], got {self.config.gamma}")

        self.gamma = self.config.gamma
        self.linucb = linucb_model or LinUCBValueModel()
        self.transition_model = transition_model or ActionConditionalTransitionModel()
        self.cost_calculator = cost_calculator or ActionCostCalculator()
        self.heuristic_model = heuristic_model or EscalateHeuristicModel()
        self.context_builder = context_builder or ContextBuilder()

    def _calculate_q1_base(
        self,
        action: ActionType,
        case: RecoveryCase,
        customer: CustomerProfile,
        context: ContextFeatures,
    ) -> tuple[float, EstimationMethod]:
        """Calculate Q1 base value without any exploration bonus."""
        if action == ActionType.STOP:
            return 0.0, EstimationMethod.HEURISTIC
        elif action == ActionType.ESCALATE:
            score = self.heuristic_model.predict_heuristic_q(case, customer)
            return float(score), EstimationMethod.HEURISTIC
        elif is_supported_action(action):
            q1 = self.linucb.predict_q1(context, action)
            return float(q1), EstimationMethod.CONTEXTUAL_BANDIT
        else:
            raise ValueError(f"Unknown action {action}")

    def _calculate_future_v1_lookahead(
        self,
        next_state: RecoveryNextState,
        case: RecoveryCase,
        customer: CustomerProfile,
        allowed_actions: List[ActionType],
    ) -> float:
        """Compute base operational policy value V1(s') = max_a' Q1(s', a').
        CRITICAL: Strictly evaluates base Q1 without any LinUCB exploration bonus B(s', a').
        """
        # Terminal states have 0 future value
        if next_state in {RecoveryNextState.RECOVERED, RecoveryNextState.UNRECOVERABLE}:
            return 0.0

        # STILL_AT_RISK state: simulate next-day context (aging + 1 day)
        simulated_case = case.model_copy(deep=True)
        simulated_case.days_overdue += 1
        simulated_case.days_waiting += 1
        sim_context = self.context_builder.build_context(simulated_case, customer)

        # Candidate future actions from allowed set (excluding STOP for positive lookahead value)
        future_candidates = [a for a in allowed_actions if a != ActionType.STOP]
        if not future_candidates:
            return 0.0

        q1_values: List[float] = []
        for a_prime in future_candidates:
            if is_supported_action(a_prime):
                # Pure base Q1 - NO EXPLORATION BONUS
                q1 = self.linucb.predict_q1(sim_context, a_prime)
                q1_values.append(q1)
            elif a_prime == ActionType.ESCALATE:
                q_esc = self.heuristic_model.predict_heuristic_q(simulated_case, customer)
                q1_values.append(q_esc)

        return float(max(q1_values)) if q1_values else 0.0

    def evaluate_action_q2(
        self,
        action: ActionType,
        case: RecoveryCase,
        customer: CustomerProfile,
        allowed_actions: List[ActionType],
    ) -> TwoStepScoringResult:
        """Compute full two-step sequence value Q2(s, a) for a single allowed action."""
        context = self.context_builder.build_context(case, customer)
        q1_base, est_method = self._calculate_q1_base(action, case, customer, context)

        # 1. Action cost C(s, a)
        cost = self.cost_calculator.calculate_cost(action, case)

        # 2. Transition distribution P(s' | s, a, x)
        trans_dist = self.transition_model.predict_transition(case.state, action, context)
        probs = trans_dist.probabilities

        # 3. Immediate Action Reward R(s, a) = P(RECOVERED) * amount_at_risk
        p_recovered = probs.get(RecoveryNextState.RECOVERED, 0.0)
        immediate_reward = float(p_recovered * float(case.amount_at_risk))

        # STOP action exception: Q2(STOP) = 0.0
        if action == ActionType.STOP:
            return TwoStepScoringResult(
                action=ActionType.STOP,
                estimation_method=EstimationMethod.HEURISTIC,
                q1_base_value=0.0,
                immediate_reward=0.0,
                action_cost=0.0,
                transition_probabilities=probs,
                future_v1_values={s: 0.0 for s in RecoveryNextState},
                expected_future_value=0.0,
                discounted_future_value=0.0,
                q2_sequence_value=0.0,
            )

        # 4. Lookahead future values V1(s') = max_a' Q1(s', a')
        future_v1_map: Dict[RecoveryNextState, float] = {}
        for s_prime in RecoveryNextState:
            v1_val = self._calculate_future_v1_lookahead(
                next_state=s_prime,
                case=case,
                customer=customer,
                allowed_actions=allowed_actions,
            )
            future_v1_map[s_prime] = float(v1_val)

        # 5. Expected future value E[V1] = sum_s' P(s' | s, a) * V1(s')
        expected_future = sum(probs[s] * future_v1_map[s] for s in RecoveryNextState)
        discounted_future = float(self.gamma * expected_future)

        # 6. Full Q2(s, a) = R(s, a) + gamma * E[V1] - C(s, a)
        q2_val = float(immediate_reward + discounted_future - cost)

        return TwoStepScoringResult(
            action=action,
            estimation_method=est_method,
            q1_base_value=q1_base,
            immediate_reward=immediate_reward,
            action_cost=cost,
            transition_probabilities=probs,
            future_v1_values=future_v1_map,
            expected_future_value=expected_future,
            discounted_future_value=discounted_future,
            q2_sequence_value=q2_val,
        )

    def evaluate_allowed_actions(
        self,
        allowed_actions: List[ActionType],
        case: RecoveryCase,
        customer: CustomerProfile,
    ) -> Dict[ActionType, TwoStepScoringResult]:
        """Evaluate Q2 sequence values strictly for all compliance-approved allowed actions."""
        results: Dict[ActionType, TwoStepScoringResult] = {}
        for action in allowed_actions:
            results[action] = self.evaluate_action_q2(
                action=action,
                case=case,
                customer=customer,
                allowed_actions=allowed_actions,
            )
        return results
