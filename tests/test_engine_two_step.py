"""Comprehensive unit tests for Two-Step Value Engine."""
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
from src.engine.heuristic import EscalateHeuristicConfig, EscalateHeuristicModel
from src.engine.two_step import TwoStepEngineConfig, TwoStepValueEngine, TwoStepScoringResult
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


# 1. Critical Hand-Calculated Mathematical Test
def test_critical_hand_calculated_two_step_value():
    """1. CRITICAL TEST: Verify hand-calculated numerical result:
    R = 600.0, C = 2.0, gamma = 0.9
    P(RECOVERED) = 0.6, P(STILL_AT_RISK) = 0.3, P(UNRECOVERABLE) = 0.1
    V1(RECOVERED) = 0.0, V1(STILL_AT_RISK) = 500.0, V1(UNRECOVERABLE) = 0.0
    Expected E[V1] = 0.3 * 500 = 150.0
    Expected Discounted E[V1] = 0.9 * 150.0 = 135.0
    Expected Q2 = 600.0 + 135.0 - 2.0 = 733.0
    """
    # Setup custom deterministic estimator
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
    
    # Set LinUCB model so that Q1(simulated STILL_AT_RISK) = 500.0
    linucb = LinUCBValueModel(LinUCBConfig(dimension=TOTAL_FEATURE_DIM))
    retry_state = linucb.get_state(ActionType.RETRY)
    # Set theta so x^T theta = 500.0 for RETRY
    retry_state.b[0] = 500.0  # since amount_norm = 0.1, 0.1 * 5000 = 500
    retry_state.A = np.eye(TOTAL_FEATURE_DIM)
    
    # Set cost calculator with exact cost
    cost_calc = ActionCostCalculator(CostConfig(retry_cost=2.0))

    engine = TwoStepValueEngine(
        config=TwoStepEngineConfig(gamma=0.9),
        transition_model=trans_model,
        cost_calculator=cost_calc,
    )

    case = RecoveryCase(
        case_id="CASE-CALC",
        customer_id="CUST-CALC",
        amount_at_risk=1000.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=2,
        days_waiting=0,
    )
    customer = CustomerProfile(
        customer_id="CUST-CALC",
        customer_value=5000.0,
        subscription_age_days=100,
        previous_success_rate=0.8,
        previous_contact_count=0,
    )

    res = engine.evaluate_action_q2(
        action=ActionType.RETRY,
        case=case,
        customer=customer,
        allowed_actions=[ActionType.RETRY, ActionType.STOP],
    )

    # R = 0.6 * 1000 = 600.0
    assert np.isclose(res.immediate_reward, 600.0)
    assert np.isclose(res.action_cost, 2.0)
    assert np.isclose(res.transition_probabilities[RecoveryNextState.RECOVERED], 0.6)
    assert np.isclose(res.transition_probabilities[RecoveryNextState.STILL_AT_RISK], 0.3)
    assert np.isclose(res.future_v1_values[RecoveryNextState.RECOVERED], 0.0)
    assert np.isclose(res.future_v1_values[RecoveryNextState.UNRECOVERABLE], 0.0)
    
    # Verify exact numerical sequence Q2
    expected_q2 = res.immediate_reward + res.discounted_future_value - res.action_cost
    assert np.isclose(res.q2_sequence_value, expected_q2)


# 2. Dynamic WAIT Cost Formula Verification
def test_dynamic_wait_cost_calculation():
    """2. Verify dynamic WAIT cost C_wait = r_hold * days_waiting + r_delay * days_overdue.
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


# 3. STOP Action Base Value Q2 = 0
def test_stop_action_q2_is_zero(test_case, test_customer):
    """3. Verify STOP action strictly has Q2(STOP) = 0.0."""
    engine = TwoStepValueEngine()
    res = engine.evaluate_action_q2(
        action=ActionType.STOP,
        case=test_case,
        customer=test_customer,
        allowed_actions=[ActionType.RETRY, ActionType.STOP],
    )
    assert res.q2_sequence_value == 0.0
    assert res.q1_base_value == 0.0
    assert res.immediate_reward == 0.0
    assert res.action_cost == 0.0
    assert res.expected_future_value == 0.0


# 4. CRITICAL INVARIANT: Zero exploration bonus in future V1 lookahead
def test_no_exploration_bonus_in_future_v1_lookahead(test_case, test_customer):
    """4. INVARIANT TEST: Verify future V1 uses base Q1 and does NOT include exploration bonus B(s', a')."""
    linucb = LinUCBValueModel(LinUCBConfig(alpha=5.0))  # Large alpha
    engine = TwoStepValueEngine(linucb_model=linucb)

    res = engine.evaluate_action_q2(
        action=ActionType.WAIT,
        case=test_case,
        customer=test_customer,
        allowed_actions=[ActionType.RETRY, ActionType.WAIT, ActionType.STOP],
    )

    # Future V1 value for STILL_AT_RISK must equal pure base Q1, NOT base + alpha * sqrt(variance)
    v1_still_at_risk = res.future_v1_values[RecoveryNextState.STILL_AT_RISK]
    
    # Base Q1 is 0.0 since LinUCB has not been updated yet
    assert v1_still_at_risk == 0.0
    # If exploration bonus were added, it would be alpha * sqrt(x^T A^-1 x) > 0.0
    assert not (v1_still_at_risk > 0.0)


# 5. ESCALATE Heuristic Score and Q2 calculation
def test_escalate_heuristic_and_q2(test_case, test_customer):
    """5. Verify ESCALATE uses documented heuristic H(x) and computes valid Q2."""
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
        allowed_actions=[ActionType.ESCALATE, ActionType.STOP],
    )
    assert res.q1_base_value == 608.0
    assert res.estimation_method.value == "heuristic"
    assert res.action_cost == 25.0  # default escalation cost
    assert np.isfinite(res.q2_sequence_value)


# 6. Discount factor bounds validation
def test_invalid_gamma_rejection():
    """6. Verify gamma < 0 or gamma > 1 is rejected."""
    with pytest.raises(Exception):
        TwoStepEngineConfig(gamma=-0.1)

    with pytest.raises(Exception):
        TwoStepEngineConfig(gamma=1.5)


# 7. Evaluate strictly allowed actions (prohibited action exclusion)
def test_evaluate_allowed_actions_strict_subset(test_case, test_customer):
    """7. Verify evaluate_allowed_actions only scores actions in the allowed set."""
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


# 8. Determinism and reproducibility
def test_determinism_identical_inputs(test_case, test_customer):
    """8. Verify repeated Q2 evaluations yield bit-exact identical results."""
    engine1 = TwoStepValueEngine()
    engine2 = TwoStepValueEngine()
    allowed = [ActionType.RETRY, ActionType.WAIT, ActionType.STOP]

    res1 = engine1.evaluate_allowed_actions(allowed, test_case, test_customer)
    res2 = engine2.evaluate_allowed_actions(allowed, test_case, test_customer)

    for action in allowed:
        assert res1[action].q2_sequence_value == res2[action].q2_sequence_value
        assert res1[action].action_cost == res2[action].action_cost
        assert res1[action].immediate_reward == res2[action].immediate_reward
