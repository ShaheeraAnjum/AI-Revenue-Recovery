"""Comprehensive unit tests for Final Action Decision Engine covering all Phase 6 criteria."""
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
from src.policy.engine import PolicyEngine, PolicyConfig
from src.models.linucb import LinUCBValueModel, LinUCBConfig
from src.engine.two_step import TwoStepValueEngine, TwoStepEngineConfig
from src.engine.decision import DecisionEngine, DecisionEngineConfig, DecisionResult, CANONICAL_TIE_BREAKER_PRIORITY


@pytest.fixture
def standard_customer() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-DEC-1",
        customer_value=6000.0,
        subscription_age_days=150,
        previous_success_rate=0.8,
        previous_contact_count=0,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
        opt_in_email=True,
        opt_in_sms=True,
    )


@pytest.fixture
def standard_case() -> RecoveryCase:
    return RecoveryCase(
        case_id="CASE-DEC-1",
        customer_id="CUST-DEC-1",
        amount_at_risk=1500.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=3,
        days_waiting=0,
        state=CaseState.ACTIVE,
    )


# 1 & 2. Policy filtering occurs before scoring; prohibited actions cannot win
def test_policy_filtering_precedes_scoring(standard_case, standard_customer):
    """1 & 2. Verify prohibited action cannot be evaluated or win even if it would have a high score."""
    policy_engine = PolicyEngine(PolicyConfig(max_retries_per_case=2))
    standard_case.retry_attempt_count = 2  # Prohibits RETRY

    decision_engine = DecisionEngine(policy_engine=policy_engine)
    result = decision_engine.decide(standard_case, standard_customer)

    assert ActionType.RETRY not in result.allowed_actions
    assert ActionType.RETRY in result.prohibited_actions
    assert result.selected_action != ActionType.RETRY
    assert ActionType.RETRY not in result.final_q_values


# 3 & 4. Correct Q2 + B calculation with explicit square root exploration bonus
def test_correct_q2_plus_b_calculation(standard_case, standard_customer):
    """3 & 4. Verify final score = Q2 + alpha * sqrt(x^T A^-1 x) for supported actions."""
    linucb = LinUCBValueModel(LinUCBConfig(alpha=2.0, dimension=TOTAL_FEATURE_DIM))
    decision_engine = DecisionEngine(
        config=DecisionEngineConfig(alpha=2.0),
        linucb_model=linucb,
    )

    result = decision_engine.decide(standard_case, standard_customer)

    for action in result.allowed_actions:
        q2 = result.q2_values[action]
        bonus = result.exploration_bonuses[action]
        final_q = result.final_q_values[action]
        
        assert np.isclose(final_q, q2 + bonus)
        if action in {ActionType.RETRY, ActionType.PAYMENT_UPDATE, ActionType.REMINDER, ActionType.WAIT}:
            assert bonus > 0.0


# 5 & 6 & 7. STOP participates in argmax and wins when all other scores are negative
def test_stop_wins_when_all_other_actions_negative(standard_case, standard_customer):
    """5, 6, 7. Verify STOP (score=0.0, bonus=0.0) wins when all other recovery actions have negative value."""
    from src.engine.costs import ActionCostCalculator, CostConfig
    from src.engine.rewards import ActionRewardCalculator, RewardConfig

    high_cost = ActionCostCalculator(CostConfig(
        retry_cost=5000.0,
        payment_update_cost=5000.0,
        reminder_cost=5000.0,
        escalation_cost=5000.0,
        r_hold=1000.0,
        r_delay=1000.0,
    ))
    zero_reward = ActionRewardCalculator(RewardConfig(
        retry_yield_factor=0.0,
        payment_update_yield_factor=0.0,
        reminder_yield_factor=0.0,
        escalation_yield_factor=0.0,
        wait_yield_factor=0.0,
    ))
    
    two_step = TwoStepValueEngine(cost_calculator=high_cost, reward_calculator=zero_reward)
    decision_engine = DecisionEngine(
        config=DecisionEngineConfig(alpha=0.0),
        two_step_engine=two_step,
    )

    result = decision_engine.decide(standard_case, standard_customer)

    assert result.selected_action == ActionType.STOP
    assert result.final_q_values[ActionType.STOP] == 0.0
    assert result.exploration_bonuses[ActionType.STOP] == 0.0
    
    for a in result.allowed_actions:
        if a != ActionType.STOP:
            assert result.final_q_values[a] < 0.0


# 8. ESCALATE has no LinUCB exploration bonus
def test_escalate_has_zero_exploration_bonus(standard_case, standard_customer):
    """8. Verify restricted action ESCALATE has strictly bonus=0.0 and estimation_method='heuristic'."""
    decision_engine = DecisionEngine(config=DecisionEngineConfig(alpha=3.0))
    result = decision_engine.decide(standard_case, standard_customer)

    if ActionType.ESCALATE in result.allowed_actions:
        assert result.exploration_bonuses[ActionType.ESCALATE] == 0.0
        assert result.estimation_methods[ActionType.ESCALATE].value == "heuristic"


# 9. Explicit forced tie-breaking verification
def test_forced_tie_breaking_canonical_priority(standard_case, standard_customer):
    """9. Verify deterministic tie-breaking priority when multiple actions have exact equal final scores.
    Canonical priority: RETRY > PAYMENT_UPDATE > REMINDER > WAIT > ESCALATE > STOP.
    """
    from src.engine.two_step import TwoStepScoringResult
    from src.audit.schema import EstimationMethod
    from src.models.transition import RecoveryNextState

    class ForcedTieTwoStepEngine(TwoStepValueEngine):
        def evaluate_allowed_actions(self, allowed_actions, case, customer):
            return {
                a: TwoStepScoringResult(
                    action=a,
                    estimation_method=EstimationMethod.CONTEXTUAL_BANDIT if a != ActionType.ESCALATE else EstimationMethod.HEURISTIC,
                    q1_base_value=100.0,
                    immediate_reward=100.0,
                    action_cost=0.0,
                    transition_probabilities={RecoveryNextState.RECOVERED: 1.0, RecoveryNextState.STILL_AT_RISK: 0.0, RecoveryNextState.UNRECOVERABLE: 0.0},
                    future_v1_values={RecoveryNextState.RECOVERED: 0.0, RecoveryNextState.STILL_AT_RISK: 0.0, RecoveryNextState.UNRECOVERABLE: 0.0},
                    expected_future_value=0.0,
                    discounted_future_value=0.0,
                    q2_sequence_value=100.0,
                )
                for a in allowed_actions
            }

    # With alpha=0.0, all actions have final_q = 100.0
    tie_engine = DecisionEngine(
        config=DecisionEngineConfig(alpha=0.0),
        two_step_engine=ForcedTieTwoStepEngine(),
    )

    # 1. RETRY and PAYMENT_UPDATE tie -> RETRY wins
    res1 = tie_engine.decide(standard_case, standard_customer, candidate_actions=[ActionType.RETRY, ActionType.PAYMENT_UPDATE])
    assert res1.final_q_values[ActionType.RETRY] == res1.final_q_values[ActionType.PAYMENT_UPDATE]
    assert res1.selected_action == ActionType.RETRY

    # 2. PAYMENT_UPDATE and REMINDER tie -> PAYMENT_UPDATE wins
    res2 = tie_engine.decide(standard_case, standard_customer, candidate_actions=[ActionType.PAYMENT_UPDATE, ActionType.REMINDER])
    assert res2.final_q_values[ActionType.PAYMENT_UPDATE] == res2.final_q_values[ActionType.REMINDER]
    assert res2.selected_action == ActionType.PAYMENT_UPDATE

    # 3. REMINDER and WAIT tie -> REMINDER wins
    res3 = tie_engine.decide(standard_case, standard_customer, candidate_actions=[ActionType.REMINDER, ActionType.WAIT])
    assert res3.selected_action == ActionType.REMINDER

    # 4. WAIT and ESCALATE tie -> WAIT wins
    res4 = tie_engine.decide(standard_case, standard_customer, candidate_actions=[ActionType.WAIT, ActionType.ESCALATE])
    assert res4.selected_action == ActionType.WAIT


# 10. Empty candidate actions integrity: NO injected STOP
def test_empty_candidate_actions_not_injected_stop(standard_case, standard_customer):
    """10. CRITICAL: Verify candidate_actions=[] produces empty allowed set and does NOT inject STOP."""
    decision_engine = DecisionEngine()
    result = decision_engine.decide(
        case=standard_case,
        customer=standard_customer,
        candidate_actions=[],
    )

    assert result.allowed_actions == []
    assert result.q2_values == {}
    assert result.exploration_bonuses == {}
    assert result.final_q_values == {}
    assert result.selected_action is None
    assert result.idempotency_key is None
    assert result.audit_record.allowed_actions == []
    assert result.audit_record.selected_action is None


# 11 & 12 & 13. Audit record and 8 version dimensions propagation
def test_audit_record_population_and_all_8_versions(standard_case, standard_customer):
    """11-13. Verify full audit record is populated with all 8 mandatory version dimensions."""
    decision_engine = DecisionEngine()
    result = decision_engine.decide(
        case=standard_case,
        customer=standard_customer,
        decision_id="DEC-TEST-1234",
        random_seed=42,
    )

    audit = result.audit_record
    assert audit.decision_id == "DEC-TEST-1234"
    assert audit.case_id == standard_case.case_id
    assert audit.customer_id == standard_customer.customer_id
    assert audit.selected_action == result.selected_action
    assert audit.random_seed == 42

    # Verify all 8 version dimensions
    assert audit.policy_version == "policy_v5.0.0"
    assert audit.value_model_version == "linucb_v5.0.0"
    assert audit.transition_model_version == "trans_v5.0.0"
    assert audit.propensity_model_version == "prop_v5.0.0"
    assert audit.fairness_policy_version == "fair_v5.0.0"
    assert audit.message_policy_version == "msg_v5.0.0"
    assert audit.feature_schema_version == "v1.1.0"
    assert audit.exploration_config_version == "exp_v5.0.0"


# 14. Idempotency key preservation
def test_idempotency_key_preservation(standard_case, standard_customer):
    """14. Verify 4-tuple key structure: case_id:decision_id:action:attempt."""
    decision_engine = DecisionEngine()
    result = decision_engine.decide(
        case=standard_case,
        customer=standard_customer,
        decision_id="DEC-IDEMP-01",
    )

    key_parts = result.idempotency_key.split(":")
    assert len(key_parts) == 4
    assert key_parts[0] == standard_case.case_id
    assert key_parts[1] == "DEC-IDEMP-01"
    assert key_parts[2] == result.selected_action.value
    assert key_parts[3] == "1"
