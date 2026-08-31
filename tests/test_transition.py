"""Comprehensive unit tests for Action-Conditional Transition Model."""
import numpy as np
import pytest
from src.domain.actions import ActionType
from src.domain.case import CaseState
from src.context.schema import TOTAL_FEATURE_DIM
from src.models.transition import (
    RecoveryNextState,
    TransitionDistribution,
    TransitionModelConfig,
    ActionConditionalTransitionModel,
    DEFAULT_TRANSITION_MODEL_VERSION,
)


# 1. State space encoding
def test_state_space_enum_encoding():
    """1. Verify finite next-state representation contains RECOVERED, STILL_AT_RISK, UNRECOVERABLE."""
    states = {s.value for s in RecoveryNextState}
    assert states == {"RECOVERED", "STILL_AT_RISK", "UNRECOVERABLE"}


# 2 & 3 & 4. Action-conditioned prediction and probability validity (finite, >=0, sum=1)
def test_action_conditional_prediction_and_validity():
    """2, 3, 4. Verify P(s' | s, a) produces valid probability distributions summing to 1.0."""
    model = ActionConditionalTransitionModel()
    context = np.zeros(TOTAL_FEATURE_DIM)
    context[0] = 0.5  # amount_at_risk_norm
    context[1] = 0.2  # days_overdue_norm
    context[4] = 0.8  # previous_success_rate

    for action in ActionType:
        dist = model.predict_transition(CaseState.ACTIVE, action, context)
        assert isinstance(dist, TransitionDistribution)
        assert dist.action == action
        assert dist.transition_model_version == DEFAULT_TRANSITION_MODEL_VERSION

        total = sum(dist.probabilities.values())
        assert np.isclose(total, 1.0, atol=1e-5)

        for state in RecoveryNextState:
            p = dist.get_probability(state)
            assert np.isfinite(p)
            assert 0.0 <= p <= 1.0


# 5. CRITICAL TEST: Different actions produce different distributions
def test_action_differentiation_distributions():
    """5. Verify that different actions produce distinctly different transition distributions for identical state/context."""
    model = ActionConditionalTransitionModel()
    context = np.zeros(TOTAL_FEATURE_DIM)
    context[0] = 0.4
    context[1] = 0.3
    context[4] = 0.7

    dist_retry = model.predict_transition(CaseState.ACTIVE, ActionType.RETRY, context)
    dist_reminder = model.predict_transition(CaseState.ACTIVE, ActionType.REMINDER, context)
    dist_wait = model.predict_transition(CaseState.ACTIVE, ActionType.WAIT, context)
    dist_escalate = model.predict_transition(CaseState.ACTIVE, ActionType.ESCALATE, context)

    # Prove P(RECOVERED | RETRY) != P(RECOVERED | REMINDER)
    assert not np.isclose(
        dist_retry.get_probability(RecoveryNextState.RECOVERED),
        dist_reminder.get_probability(RecoveryNextState.RECOVERED),
    )

    # Prove P(STILL_AT_RISK | WAIT) is distinct and higher than active actions
    assert dist_wait.get_probability(RecoveryNextState.STILL_AT_RISK) > dist_retry.get_probability(RecoveryNextState.STILL_AT_RISK)

    # Prove P(RECOVERED | ESCALATE) is distinct from WAIT
    assert not np.isclose(
        dist_escalate.get_probability(RecoveryNextState.RECOVERED),
        dist_wait.get_probability(RecoveryNextState.RECOVERED),
    )


# 6. Determinism and reproducibility
def test_determinism_identical_inputs():
    """6. Verify identical inputs produce bit-exact identical transition distributions."""
    model1 = ActionConditionalTransitionModel()
    model2 = ActionConditionalTransitionModel()
    context = np.linspace(0.1, 0.9, TOTAL_FEATURE_DIM)

    dist1 = model1.predict_transition(CaseState.ACTIVE, ActionType.PAYMENT_UPDATE, context)
    dist2 = model2.predict_transition(CaseState.ACTIVE, ActionType.PAYMENT_UPDATE, context)

    assert dist1.probabilities == dist2.probabilities


# 7 & 8 & 9. Supported actions, ESCALATE, and STOP handling
def test_all_actions_handled_appropriately():
    """7, 8, 9. Verify supported actions, ESCALATE, and STOP are all handled correctly."""
    model = ActionConditionalTransitionModel()
    context = np.zeros(TOTAL_FEATURE_DIM)

    # STOP deterministically transitions to UNRECOVERABLE with probability 1.0
    dist_stop = model.predict_transition(CaseState.ACTIVE, ActionType.STOP, context)
    assert dist_stop.get_probability(RecoveryNextState.UNRECOVERABLE) == 1.0
    assert dist_stop.get_probability(RecoveryNextState.RECOVERED) == 0.0
    assert dist_stop.get_probability(RecoveryNextState.STILL_AT_RISK) == 0.0

    # ESCALATE has high recovery probability
    dist_esc = model.predict_transition(CaseState.ACTIVE, ActionType.ESCALATE, context)
    assert dist_esc.get_probability(RecoveryNextState.RECOVERED) > 0.5


# 10 & 11. Invalid and non-finite probability / input rejection
def test_invalid_input_rejection():
    """10 & 11. Verify non-finite and dimension mismatched inputs are rejected."""
    model = ActionConditionalTransitionModel()
    
    # Dimension mismatch
    with pytest.raises(ValueError, match="dimension mismatch"):
        model.predict_transition(CaseState.ACTIVE, ActionType.RETRY, np.ones(15))

    # NaN / Inf in context
    bad_ctx = np.ones(TOTAL_FEATURE_DIM)
    bad_ctx[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        model.predict_transition(CaseState.ACTIVE, ActionType.RETRY, bad_ctx)


# 12. Version propagation
def test_transition_version_propagation():
    """12. Verify transition_model_version is propagated in the result."""
    config = TransitionModelConfig(transition_model_version="trans_v5.2.0-custom")
    model = ActionConditionalTransitionModel(config)
    dist = model.predict_transition(CaseState.ACTIVE, ActionType.WAIT, np.zeros(TOTAL_FEATURE_DIM))
    assert dist.transition_model_version == "trans_v5.2.0-custom"


# 13. Future Q2 expectation compatibility test
def test_future_two_step_q2_expectation_compatibility():
    """15. Verify distribution structure directly enables calculating sum_{s'} P(s' | s, a) * V1(s')."""
    model = ActionConditionalTransitionModel()
    context = np.zeros(TOTAL_FEATURE_DIM)
    dist = model.predict_transition(CaseState.ACTIVE, ActionType.RETRY, context)

    # Mock future values V1(s') for each next state
    mock_v1 = {
        RecoveryNextState.RECOVERED: 1000.0,
        RecoveryNextState.STILL_AT_RISK: 400.0,
        RecoveryNextState.UNRECOVERABLE: 0.0,
    }

    # Calculate expectation E[V1] = sum_{s'} P(s' | s, a) * V1(s')
    expected_future_value = sum(
        dist.get_probability(next_state) * mock_v1[next_state]
        for next_state in RecoveryNextState
    )

    assert np.isfinite(expected_future_value)
    assert expected_future_value > 0.0
