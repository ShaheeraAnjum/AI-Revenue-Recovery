"""Comprehensive unit tests for Action-Conditional Transition Model."""
import numpy as np
import pytest
from src.domain.actions import ActionType
from src.domain.case import CaseState
from src.context.schema import TOTAL_FEATURE_DIM, CANONICAL_FEATURE_NAMES
from src.models.transition import (
    RecoveryNextState,
    TransitionEstimationMethod,
    TransitionDistribution,
    TransitionModelConfig,
    BaseTransitionEstimator,
    CalibratedPriorTransitionEstimator,
    ActionConditionalTransitionModel,
    DEFAULT_TRANSITION_MODEL_VERSION,
)


# 1. State space encoding and terminal mapping
def test_state_space_enum_encoding():
    """1. Verify finite next-state representation contains RECOVERED, STILL_AT_RISK, UNRECOVERABLE."""
    states = {s.value for s in RecoveryNextState}
    assert states == {"RECOVERED", "STILL_AT_RISK", "UNRECOVERABLE"}


# 2 & 3 & 4. Action-conditioned prediction, probability validity, and estimation method
def test_action_conditional_prediction_and_validity():
    """2, 3, 4. Verify P(s' | s, a) produces valid probability distributions summing to 1.0 with explicit provenance."""
    model = ActionConditionalTransitionModel()
    context = np.zeros(TOTAL_FEATURE_DIM)
    context[0] = 0.5  # amount_at_risk_norm
    context[1] = 0.2  # days_overdue_norm
    context[4] = 0.8  # previous_success_rate

    for action in ActionType:
        dist = model.predict_transition(CaseState.ACTIVE, action, context)
        assert isinstance(dist, TransitionDistribution)
        assert dist.current_state == CaseState.ACTIVE
        assert dist.action == action
        assert dist.transition_model_version == DEFAULT_TRANSITION_MODEL_VERSION
        assert dist.estimation_method == TransitionEstimationMethod.CALIBRATED_PRIOR

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


# 6. Named context features without magic indices
def test_named_context_feature_extraction():
    """6. Verify feature extraction uses canonical schema names rather than unexplained raw indices."""
    model = ActionConditionalTransitionModel()
    context = np.zeros(TOTAL_FEATURE_DIM)
    context[0] = 0.9  # amount_at_risk_norm
    context[1] = 0.5  # days_overdue_norm

    named = model._extract_named_features(context)
    assert "amount_at_risk_norm" in named
    assert "days_overdue_norm" in named
    assert named["amount_at_risk_norm"] == 0.9
    assert named["days_overdue_norm"] == 0.5


# 7. Estimator Interface and Pluggability
def test_custom_estimator_interface_pluggability():
    """7. Verify BaseTransitionEstimator interface allows swapping in custom learned estimators seamlessly."""
    class CustomLearnedEstimator(BaseTransitionEstimator):
        @property
        def estimation_method(self) -> TransitionEstimationMethod:
            return TransitionEstimationMethod.LEARNED

        def estimate(self, state, action, named_features):
            return {
                RecoveryNextState.RECOVERED: 0.7,
                RecoveryNextState.STILL_AT_RISK: 0.2,
                RecoveryNextState.UNRECOVERABLE: 0.1,
            }

    custom_model = ActionConditionalTransitionModel(estimator=CustomLearnedEstimator())
    dist = custom_model.predict_transition(CaseState.ACTIVE, ActionType.RETRY, np.zeros(TOTAL_FEATURE_DIM))
    
    assert dist.estimation_method == TransitionEstimationMethod.LEARNED
    assert np.isclose(dist.get_probability(RecoveryNextState.RECOVERED), 0.7)


# 8. STOP action determinism
def test_stop_action_terminal_transition():
    """8. Verify STOP deterministically transitions to UNRECOVERABLE with probability 1.0."""
    model = ActionConditionalTransitionModel()
    dist = model.predict_transition(CaseState.ACTIVE, ActionType.STOP, np.zeros(TOTAL_FEATURE_DIM))

    assert dist.get_probability(RecoveryNextState.UNRECOVERABLE) == 1.0
    assert dist.get_probability(RecoveryNextState.RECOVERED) == 0.0
    assert dist.get_probability(RecoveryNextState.STILL_AT_RISK) == 0.0


# 9. Invalid and non-finite probability / input rejection
def test_invalid_input_rejection():
    """9. Verify non-finite and dimension mismatched inputs are rejected."""
    model = ActionConditionalTransitionModel()
    
    with pytest.raises(ValueError, match="dimension mismatch"):
        model.predict_transition(CaseState.ACTIVE, ActionType.RETRY, np.ones(15))

    bad_ctx = np.ones(TOTAL_FEATURE_DIM)
    bad_ctx[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        model.predict_transition(CaseState.ACTIVE, ActionType.RETRY, bad_ctx)


# 10. Future Q2 expectation compatibility test
def test_future_two_step_q2_expectation_compatibility():
    """10. Verify distribution structure directly enables calculating sum_{s'} P(s' | s, a) * V1(s')."""
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
