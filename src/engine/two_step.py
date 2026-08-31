"""Two-step sequence value engine implementing Q2(s, a) = R + gamma * sum(P * V1) - C."""
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.domain.actions import ActionType, is_supported_action
from src.domain.case import RecoveryCase, CustomerProfile, CaseState
from src.context.schema import ContextFeatures
from src.context.builder import ContextBuilder
from src.policy.engine import PolicyEngine
from src.models.linucb import LinUCBValueModel
from src.models.transition import (
    ActionConditionalTransitionModel,
    RecoveryNextState,
    TransitionDistribution,
)
from src.engine.costs import ActionCostCalculator, CostConfig
from src.engine.rewards import ActionRewardCalculator, RewardConfig
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
    q1_base_value: float = Field(..., description="Diagnostic base value Q1(s, a) for current state (NOT added to Q2)")
    immediate_reward: float = Field(..., description="Action-dependent immediate reward R(s, a) from RewardModel")
    action_cost: float = Field(..., description="Action-specific cost C(s, a)")
    transition_probabilities: Dict[RecoveryNextState, float] = Field(
        ..., description="Transition probabilities P(s' | s, a) from TransitionModel"
    )
    future_v1_values: Dict[RecoveryNextState, float] = Field(
        ..., description="Lookahead base policy values V1(s') = max_{a' in A_allowed(s')} Q1(s', a') (strictly NO exploration bonus)"
    )
    expected_future_value: float = Field(..., description="Expectation sum_s' P(s' | s, a) * V1(s')")
    discounted_future_value: float = Field(..., description="gamma * sum_s' P(s' | s, a) * V1(s')")
    q2_sequence_value: float = Field(..., description="Full sequence value Q2(s, a) = R(s, a) + gamma * E[V1] - C(s, a)")


class TwoStepValueEngine:
    """Calculates Two-Step Sequence Value Q2(s, a) across compliance-approved allowed actions.
    Enforces strict mathematical separation:
    - Base Q1 is used for future lookahead states V1(s') = max_{a' in A_allowed(s')} Q1(s', a') (zero future exploration).
    - Future state evaluates its OWN policy-approved action set A_allowed(s') via PolicyEngine.
    - STOP action participates in future action selection (Q(STOP)=0.0).
    - Current exploration bonus B(x, a) is NOT computed here (deferred to decision layer).
    """

    def __init__(
        self,
        config: Optional[TwoStepEngineConfig] = None,
        linucb_model: Optional[LinUCBValueModel] = None,
        transition_model: Optional[ActionConditionalTransitionModel] = None,
        cost_calculator: Optional[ActionCostCalculator] = None,
        reward_calculator: Optional[ActionRewardCalculator] = None,
        heuristic_model: Optional[EscalateHeuristicModel] = None,
        policy_engine: Optional[PolicyEngine] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.config = config or TwoStepEngineConfig()
        if not (0.0 <= self.config.gamma <= 1.0):
            raise ValueError(f"Discount factor gamma must be in [0, 1], got {self.config.gamma}")

        self.gamma = self.config.gamma
        self.linucb = linucb_model or LinUCBValueModel()
        self.transition_model = transition_model or ActionConditionalTransitionModel()
        self.cost_calculator = cost_calculator or ActionCostCalculator()
        self.reward_calculator = reward_calculator or ActionRewardCalculator()
        self.heuristic_model = heuristic_model or EscalateHeuristicModel()
        self.policy_engine = policy_engine or PolicyEngine()
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

    def _construct_hypothetical_future_case(
        self,
        current_case: RecoveryCase,
        action_taken: ActionType,
        next_state: RecoveryNextState,
    ) -> RecoveryCase:
        """Construct simulated hypothetical future case state reflecting action execution and state transition."""
        future_case = current_case.model_copy(deep=True)
        if next_state == RecoveryNextState.RECOVERED:
            future_case.state = CaseState.RESOLVED_RECOVERED
        elif next_state == RecoveryNextState.UNRECOVERABLE or action_taken == ActionType.STOP:
            future_case.state = CaseState.RESOLVED_UNRECOVERABLE
        else:
            # STILL_AT_RISK: age the case and increment action execution attempt counters
            future_case.state = CaseState.ACTIVE
            future_case.days_overdue += 1
            if action_taken == ActionType.WAIT:
                future_case.days_waiting += 1
            else:
                future_case.days_waiting = 0

            if action_taken == ActionType.RETRY:
                future_case.retry_attempt_count += 1
            elif action_taken == ActionType.REMINDER:
                future_case.reminder_count += 1
            elif action_taken == ActionType.ESCALATE:
                future_case.escalation_count += 1

        return future_case

    def _calculate_future_v1_lookahead(
        self,
        next_state: RecoveryNextState,
        action_taken: ActionType,
        current_case: RecoveryCase,
        customer: CustomerProfile,
    ) -> float:
        """Compute base operational policy value V1(s') = max_{a' in A_allowed(s')} Q1(s', a').
        CRITICAL RULES:
        1. Evaluates future policy via PolicyEngine for the hypothetical future state.
        2. Strictly evaluates base Q1 without any LinUCB exploration bonus B(s', a').
        3. STOP participates in future action selection (Q(STOP)=0.0).
        """
        if next_state in {RecoveryNextState.RECOVERED, RecoveryNextState.UNRECOVERABLE}:
            return 0.0

        # Construct hypothetical future state
        future_case = self._construct_hypothetical_future_case(current_case, action_taken, next_state)
        
        # Evaluate policy engine on the hypothetical future state
        future_policy = self.policy_engine.evaluate(future_case, customer)
        future_allowed = future_policy.allowed_actions

        if not future_allowed:
            return 0.0

        sim_context = self.context_builder.build_context(future_case, customer)

        # Compute pure base Q1 for every future allowed action (including STOP)
        q1_values: List[float] = []
        for a_prime in future_allowed:
            if a_prime == ActionType.STOP:
                q1_values.append(0.0)
            elif is_supported_action(a_prime):
                # Pure base Q1 - NO EXPLORATION BONUS
                q1 = self.linucb.predict_q1(sim_context, a_prime)
                q1_values.append(q1)
            elif a_prime == ActionType.ESCALATE:
                q_esc = self.heuristic_model.predict_heuristic_q(future_case, customer)
                q1_values.append(q_esc)

        return float(max(q1_values)) if q1_values else 0.0

    def evaluate_action_q2(
        self,
        action: ActionType,
        case: RecoveryCase,
        customer: CustomerProfile,
    ) -> TwoStepScoringResult:
        """Compute full two-step sequence value Q2(s, a) for a single allowed action."""
        context = self.context_builder.build_context(case, customer)
        q1_diagnostic, est_method = self._calculate_q1_base(action, case, customer, context)

        # 1. Action cost C(s, a) from ActionCostCalculator
        cost = self.cost_calculator.calculate_cost(action, case)

        # 2. Action reward R(s, a) from ActionRewardCalculator
        immediate_reward = self.reward_calculator.calculate_reward(action, case, customer)

        # 3. Transition distribution P(s' | s, a, x) from ActionConditionalTransitionModel
        trans_dist = self.transition_model.predict_transition(case.state, action, context)
        probs = trans_dist.probabilities

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

        # 4. Lookahead future values V1(s') = max_{a' in A_allowed(s')} Q1(s', a')
        future_v1_map: Dict[RecoveryNextState, float] = {}
        for s_prime in RecoveryNextState:
            v1_val = self._calculate_future_v1_lookahead(
                next_state=s_prime,
                action_taken=action,
                current_case=case,
                customer=customer,
            )
            future_v1_map[s_prime] = float(v1_val)

        # 5. Expected future value E[V1] = sum_s' P(s' | s, a) * V1(s')
        expected_future = sum(probs[s] * future_v1_map[s] for s in RecoveryNextState)
        discounted_future = float(self.gamma * expected_future)

        # 6. Full Q2(s, a) = R(s, a) + gamma * E[V1] - C(s, a)
        # CRITICAL: q1_diagnostic is NOT added to Q2!
        q2_val = float(immediate_reward + discounted_future - cost)

        return TwoStepScoringResult(
            action=action,
            estimation_method=est_method,
            q1_base_value=q1_diagnostic,
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
            )
        return results
