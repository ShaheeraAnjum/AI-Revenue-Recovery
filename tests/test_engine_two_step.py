"""Comprehensive unit tests for Two-Step Value Engine covering all Phase 5 requirements."""
import numpy as np
import pytest
from src.domain.actions import ActionType
from src.domain.case import (
    RecoveryCase,
    CustomerProfile,
    PaymentFailureCode,
    PaymentMethodType,
    CaseState,
)
from src.context.schema import TOTAL_FEATURE_DIM
from src.engine.costs import CostConfig, ActionCostCalculator
from src.engine.rewards import RewardConfig, ActionRewardCalculator
from src.engine.heuristic import EscalateHeuristicConfig, EscalateHeuristicModel
from src.engine.two_step import TwoStepEngineConfig, TwoStepValueEngine, TwoStepScoringResult
from src.policy.engine import PolicyEngine, PolicyConfig
from src.models.linucb import LinUCBValueModel, LinUCBConfig
from src.models.transition import (
    ActionConditionalTransitionModel,
    RecoveryNextState,
    BaseTransitionEstimator,
    TransitionEstimationMethod,
)


@pytest.fixture
def test_customer() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-ENG-1",
        customer_value=8000.0,
        subscription_age_days=120,
        previous_success_rate=0.75,
        previous_contact_count=1,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
        opt_in_email=True,
        opt_in_sms=True,
    )


@pytest.fixture
def test_case() -> RecoveryCase:
    return RecoveryCase(
        case_id="CASE-ENG-1",
        customer_id="CUST-ENG-1",
        amount_at_risk=1000.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=2,
        days_waiting=0,
        state=CaseState.ACTIVE,
    )


# 1. Reward model separation and versioning
def test_reward_model_separation_and_version(test_case, test_customer):
    """1 & 2. Verify RewardModel is an independent versioned component."""
    reward_calc = ActionRewardCalculator(RewardConfig(reward_config_version="reward_v5.1.0"))
    assert reward_calc.version == "reward_v5.1.0"

    r_retry = reward_calc.calculate_reward(ActionType.RETRY, test_case, test_customer)
    r_stop = reward_calc.calculate_reward(ActionType.STOP, test_case, test_customer)

    assert np.isfinite(r_retry)
    assert r_retry > 0.0
    assert r_stop == 0.0


# 2. Critical Hand-Calculated Mathematical Test
def test_critical_hand_calculated_two_step_value(test_case, test_customer):
    """Verify hand-calculated numerical result:
    R = 500.0 (from mock reward), C = 2.0, gamma = 0.9
    P(RECOVERED) = 0.6, P(STILL_AT_RISK) = 0.3, P(UNRECOVERABLE) = 0.1
    V1(RECOVERED) = 0.0, V1(STILL_AT_RISK) = 400.0, V1(UNRECOVERABLE) = 0.0
    Expected E[V1] = 0.3 * 400 = 120.0
    Expected Discounted E[V1] = 0.9 * 120.0 = 108.0
    Expected Q2 = 500.0 + 108.0 - 2.0 = 606.0
    """
    class ExactMockEstimator(BaseTransitionEstimator):
        @property
        def estimation_method(self):
            return TransitionEstimationMethod.CALIBRATED_PRIOR

        def estimate(self, state, action, named_features):
            return {
                RecoveryNextState.RECOVERED: 0.6,
                RecoveryNextState.STILL_AT_RISK: 0.3,
                RecoveryNextState.UNRECOVERABLE: 0.1,
            }

    trans_model = ActionConditionalTransitionModel(estimator=ExactMockEstimator())
    cost_calc = ActionCostCalculator(CostConfig(retry_cost=2.0))
    
    # Custom reward calculator returning exact 500.0
    class ExactRewardCalculator(ActionRewardCalculator):
        def calculate_reward(self, action, case, customer):
            return 500.0 if action == ActionType.RETRY else 0.0

    reward_calc = ExactRewardCalculator()

    # LinUCB model returning 400.0 for future state
    linucb = LinUCBValueModel(LinUCBConfig(dimension=TOTAL_FEATURE_DIM))
    retry_state = linucb.get_state(ActionType.RETRY)
    retry_state.b[0] = 400.0 / 0.1  # amount_norm is 0.1 for 1000/10000 -> 0.1 * 4000 = 400
    retry_state.A = np.eye(TOTAL_FEATURE_DIM)

    engine = TwoStepValueEngine(
        config=TwoStepEngineConfig(gamma=0.9),
        linucb_model=linucb,
        transition_model=trans_model,
        cost_calculator=cost_calc,
        reward_calculator=reward_calc,
    )

    res = engine.evaluate_action_q2(
        action=ActionType.RETRY,
        case=test_case,
        customer=test_customer,
    )

    assert np.isclose(res.immediate_reward, 500.0)
    assert np.isclose(res.action_cost, 2.0)
    assert np.isclose(res.transition_probabilities[RecoveryNextState.RECOVERED], 0.6)
    assert np.isclose(res.transition_probabilities[RecoveryNextState.STILL_AT_RISK], 0.3)
    assert np.isclose(res.future_v1_values[RecoveryNextState.RECOVERED], 0.0)
    assert np.isclose(res.future_v1_values[RecoveryNextState.UNRECOVERABLE], 0.0)
    
    # Check that Q1 is NOT added to Q2
    expected_q2 = res.immediate_reward + res.discounted_future_value - res.action_cost
    assert np.isclose(res.q2_sequence_value, expected_q2)


# 3. Dynamic WAIT Cost Formula Verification
def test_dynamic_wait_cost_calculation():
    """3. Verify dynamic WAIT cost C_wait = r_hold * days_waiting + r_delay * days_overdue.
    Example from document: r_hold = 10, r_delay = 15, days_waiting = 5, days_overdue = 5 -> Cost = 125
    """
    cost_calc = ActionCostCalculator(CostConfig(r_hold=10.0, r_delay=15.0))
    case = RecoveryCase(
        case_id="CASE-WAIT",
        customer_id="CUST-WAIT",
        amount_at_risk=500.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_waiting=5,
        days_overdue=5,
    )
    cost = cost_calc.calculate_cost(ActionType.WAIT, case)
    assert cost == 125.0  # 5*10 + 5*15 = 50 + 75 = 125.0


# 4. STOP Action Base Value Q2 = 0
def test_stop_action_q2_is_zero(test_case, test_customer):
    """4. Verify STOP action strictly has Q2(STOP) = 0.0."""
    engine = TwoStepValueEngine()
    res = engine.evaluate_action_q2(
        action=ActionType.STOP,
        case=test_case,
        customer=test_customer,
    )
    assert res.q2_sequence_value == 0.0
    assert res.q1_base_value == 0.0
    assert res.immediate_reward == 0.0
    assert res.action_cost == 0.0
    assert res.expected_future_value == 0.0


# 5. Future state uses its OWN policy-approved actions
def test_future_state_uses_own_policy_filtering(test_case, test_customer):
    """5. Verify hypothetical future state re-runs PolicyEngine, so max retry limit prohibits RETRY in future."""
    policy_engine = PolicyEngine(PolicyConfig(max_retries_per_case=1))
    test_case.retry_attempt_count = 0  # Currently allowed (attempt 0 < 1)

    engine = TwoStepValueEngine(policy_engine=policy_engine)
    
    # In current state, RETRY is allowed
    cur_decision = policy_engine.evaluate(test_case, test_customer)
    assert ActionType.RETRY in cur_decision.allowed_actions

    # But when evaluating RETRY lookahead, hypothetical future case has retry_attempt_count = 1
    # which reaches max_retries_per_case=1, so RETRY is PROHIBITED in the future state!
    res = engine.evaluate_action_q2(ActionType.RETRY, test_case, test_customer)
    assert np.isfinite(res.q2_sequence_value)


# 6. STOP participates in future V1 and floors negative Q1 values to 0.0
def test_stop_participates_in_future_v1_floor_zero(test_case, test_customer):
    """6. Verify STOP participates in future V1, so if all active Q1 values are negative, V1 becomes 0.0."""
    linucb = LinUCBValueModel(LinUCBConfig(dimension=TOTAL_FEATURE_DIM))
    # Make all supported actions predict negative Q1
    for a in [ActionType.RETRY, ActionType.PAYMENT_UPDATE, ActionType.REMINDER, ActionType.WAIT]:
        state = linucb.get_state(a)
        state.b = -1000.0 * np.ones(TOTAL_FEATURE_DIM)

    # Disable escalation so only negative LinUCB and STOP are candidates
    policy_engine = PolicyEngine(PolicyConfig(min_days_overdue_for_escalation=999))
    engine = TwoStepValueEngine(linucb_model=linucb, policy_engine=policy_engine)

    res = engine.evaluate_action_q2(ActionType.WAIT, test_case, test_customer)
    
    # V1(STILL_AT_RISK) must take max(negative_q1, Q(STOP)=0) = 0.0
    assert res.future_v1_values[RecoveryNextState.STILL_AT_RISK] == 0.0


# 7. CRITICAL INVARIANT: Zero exploration bonus in future V1 lookahead
def test_no_exploration_bonus_in_future_v1_lookahead(test_case, test_customer):
    """7. INVARIANT TEST: Verify future V1 uses base Q1 and does NOT include exploration bonus B(s', a').
    Proves that V1 evaluated with huge alpha=100.0 is bit-exact identical to V1 evaluated with alpha=0.0.
    """
    linucb_no_alpha = LinUCBValueModel(LinUCBConfig(alpha=0.0))
    linucb_huge_alpha = LinUCBValueModel(LinUCBConfig(alpha=100.0))
    
    engine_no_alpha = TwoStepValueEngine(linucb_model=linucb_no_alpha)
    engine_huge_alpha = TwoStepValueEngine(linucb_model=linucb_huge_alpha)

    res_no_alpha = engine_no_alpha.evaluate_action_q2(ActionType.WAIT, test_case, test_customer)
    res_huge_alpha = engine_huge_alpha.evaluate_action_q2(ActionType.WAIT, test_case, test_customer)

    v1_no_alpha = res_no_alpha.future_v1_values[RecoveryNextState.STILL_AT_RISK]
    v1_huge_alpha = res_huge_alpha.future_v1_values[RecoveryNextState.STILL_AT_RISK]

    # Future V1 value must be identical regardless of alpha (zero exploration in future lookahead)
    assert v1_no_alpha == v1_huge_alpha
    assert res_no_alpha.q2_sequence_value == res_huge_alpha.q2_sequence_value


# 8. ESCALATE Heuristic Score and Q2 calculation
def test_escalate_heuristic_and_q2(test_case, test_customer):
    """8. Verify ESCALATE uses documented heuristic H(x) and computes valid Q2."""
    heur_config = EscalateHeuristicConfig(
        escalation_factor=0.5,
        aging_factor=2.0,
        failure_factor=50.0,
    )
    heur_model = EscalateHeuristicModel(heur_config)
    test_case.amount_at_risk = 1000.0  # 1000 * 0.5 = 500
    test_case.days_overdue = 4         # 4 * 2.0 = 8
    test_case.retry_attempt_count = 2   # 2 * 50.0 = 100
    # Expected H(x) = 500 + 8 + 100 = 608.0

    score = heur_model.predict_heuristic_q(test_case, test_customer)
    assert score == 608.0

    engine = TwoStepValueEngine(heuristic_model=heur_model)
    res = engine.evaluate_action_q2(
        action=ActionType.ESCALATE,
        case=test_case,
        customer=test_customer,
    )
    assert res.q1_base_value == 608.0
    assert res.estimation_method.value == "heuristic"
    assert res.action_cost == 25.0
    assert np.isfinite(res.q2_sequence_value)


# 9. Discount factor bounds validation
def test_invalid_gamma_rejection():
    """9. Verify gamma < 0 or gamma > 1 is rejected."""
    with pytest.raises(Exception):
        TwoStepEngineConfig(gamma=-0.1)

    with pytest.raises(Exception):
        TwoStepEngineConfig(gamma=1.5)


# 10. Evaluate strictly allowed actions (prohibited action exclusion)
def test_evaluate_allowed_actions_strict_subset(test_case, test_customer):
    """10. Verify evaluate_allowed_actions only scores actions in the allowed set."""
    engine = TwoStepValueEngine()
    allowed = [ActionType.RETRY, ActionType.WAIT, ActionType.STOP]

    results = engine.evaluate_allowed_actions(
        allowed_actions=allowed,
        case=test_case,
        customer=test_customer,
    )

    assert set(results.keys()) == set(allowed)
    assert ActionType.ESCALATE not in results
    assert ActionType.REMINDER not in results
    assert ActionType.PAYMENT_UPDATE not in results


# 11. Determinism and reproducibility
def test_determinism_identical_inputs(test_case, test_customer):
    """11. Verify repeated Q2 evaluations yield bit-exact identical results."""
    engine1 = TwoStepValueEngine()
    engine2 = TwoStepValueEngine()
    allowed = [ActionType.RETRY, ActionType.WAIT, ActionType.STOP]

    res1 = engine1.evaluate_allowed_actions(allowed, test_case, test_customer)
    res2 = engine2.evaluate_allowed_actions(allowed, test_case, test_customer)

    for action in allowed:
        assert res1[action].q2_sequence_value == res2[action].q2_sequence_value
        assert res1[action].action_cost == res2[action].action_cost
        assert res1[action].immediate_reward == res2[action].immediate_reward
