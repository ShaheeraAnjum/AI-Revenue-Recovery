"""Comprehensive unit tests for Contextual LinUCB Value Model covering all 20 required verification criteria."""
import numpy as np
import pytest
from src.domain.actions import ActionType, SUPPORTED_ACTIONS
from src.models.config import LinUCBConfig, DEFAULT_VALUE_MODEL_VERSION
from src.models.linucb import LinUCBValueModel, ActionLinUCBState


# 1. Initialization tests: A_a = lambda * I_d
def test_initialization_regularization_matrices():
    """1. Verify A_a = lambda * I_d for all supported actions."""
    dim = 20
    lambda_reg = 2.5
    config = LinUCBConfig(dimension=dim, lambda_reg=lambda_reg, alpha=1.2)
    model = LinUCBValueModel(config)

    assert model.dimension == 20
    assert model.alpha == 1.2
    assert model.lambda_reg == 2.5

    for action in SUPPORTED_ACTIONS:
        state = model.get_state(action)
        assert state.A.shape == (dim, dim)
        assert np.allclose(state.A, lambda_reg * np.eye(dim))


# 2. Initialization tests: b_a = 0_d
def test_initialization_b_vector_zero():
    """2. Verify b_a = 0_d for all supported actions."""
    dim = 20
    model = LinUCBValueModel(LinUCBConfig(dimension=dim))
    for action in SUPPORTED_ACTIONS:
        state = model.get_state(action)
        assert state.b.shape == (dim,)
        assert np.allclose(state.b, np.zeros(dim))


# 3. Correct theta calculation: theta_a = A_a^-1 b_a
def test_theta_calculation():
    """3. Verify theta_a is correctly solved from A_a theta = b_a."""
    dim = 2
    model = LinUCBValueModel(LinUCBConfig(dimension=dim, lambda_reg=1.0))
    state = model.get_state(ActionType.RETRY)
    
    # Initially theta = [0, 0]
    assert np.allclose(state.theta, np.zeros(dim))

    # Set known A and b: A = [[2, 1], [1, 2]], b = [3, 3]
    state.A = np.array([[2.0, 1.0], [1.0, 2.0]])
    state.b = np.array([3.0, 3.0])
    # Solution: theta = [1.0, 1.0] since 2(1)+1(1)=3
    assert np.allclose(state.theta, np.array([1.0, 1.0]))


# 4. Correct Q1 calculation: Q1(x, a) = x^T theta_a
def test_q1_base_value_calculation():
    """4. Verify Q1(x, a) = x^T theta_a accurately."""
    dim = 3
    config = LinUCBConfig(dimension=dim, lambda_reg=1.0, alpha=1.0)
    model = LinUCBValueModel(config)

    x = np.array([1.0, 2.0, 3.0])
    assert model.predict_q1(x, ActionType.RETRY) == 0.0

    state = model.get_state(ActionType.RETRY)
    state.b = np.array([2.0, 4.0, 6.0])
    # A = I_3 -> theta = [2, 4, 6] -> Q1 = 1*2 + 2*4 + 3*6 = 28.0
    assert np.isclose(model.predict_q1(x, ActionType.RETRY), 28.0)


# 5 & 6. CRITICAL MATHEMATICAL TEST: Exact LinUCB exploration formula and square-root verification
def test_critical_mathematical_formula_square_root_distinction():
    """5 & 6. CRITICAL TEST: Hand-calculated numerical verification of B(x, a) = alpha * sqrt(x^T A^-1 x).
    Explicitly distinguishes alpha * sqrt(x^T A^-1 x) from incorrect alpha * (x^T A^-1 x).
    """
    dim = 2
    lambda_reg = 2.0
    alpha = 1.5
    config = LinUCBConfig(dimension=dim, lambda_reg=lambda_reg, alpha=alpha)
    model = LinUCBValueModel(config)

    # Initial A = 2 * I_2 = [[2, 0], [0, 2]]
    # A^-1 = [[0.5, 0], [0, 0.5]]
    # x = [3.0, 4.0]
    # x^T A^-1 x = 3^2 * 0.5 + 4^2 * 0.5 = 4.5 + 8.0 = 12.5
    # sqrt(x^T A^-1 x) = sqrt(12.5) = 3.5355339059327378
    # Correct B(x, a) = 1.5 * sqrt(12.5) = 5.303300858899106
    # Incorrect B_wrong = 1.5 * 12.5 = 18.75

    x = np.array([3.0, 4.0], dtype=np.float64)
    expected_quad_form = 12.5
    expected_sqrt_term = np.sqrt(expected_quad_form)
    expected_bonus = alpha * expected_sqrt_term
    incorrect_linear_bonus = alpha * expected_quad_form

    calculated_bonus = model.compute_exploration_bonus(x, ActionType.RETRY)

    assert np.isclose(calculated_bonus, expected_bonus, rtol=1e-9)
    assert not np.isclose(calculated_bonus, incorrect_linear_bonus, rtol=1e-2)
    assert abs(calculated_bonus - 5.303300858899106) < 1e-9


# 7. Independent parameters per action
def test_independent_parameters_per_action():
    """7. Verify each action has distinct matrices."""
    model = LinUCBValueModel(LinUCBConfig(dimension=4))
    for a1 in SUPPORTED_ACTIONS:
        for a2 in SUPPORTED_ACTIONS:
            if a1 != a2:
                assert model.get_state(a1) is not model.get_state(a2)
                assert model.get_state(a1).A is not model.get_state(a2).A


# 8 & 9. Correct update of selected action while unselected actions remain unchanged
def test_update_selected_action_isolation():
    """8 & 9. Verify updating selected action updates its matrices and leaves all other actions untouched."""
    dim = 4
    config = LinUCBConfig(dimension=dim, lambda_reg=1.0)
    model = LinUCBValueModel(config)

    x = np.array([1.0, 0.5, 0.2, 0.1])
    reward = 100.0

    init_A_update = np.copy(model.get_state(ActionType.PAYMENT_UPDATE).A)
    init_b_reminder = np.copy(model.get_state(ActionType.REMINDER).b)
    init_A_wait = np.copy(model.get_state(ActionType.WAIT).A)

    model.update(x, ActionType.RETRY, reward)

    retry_state = model.get_state(ActionType.RETRY)
    assert retry_state.update_count == 1
    assert np.allclose(retry_state.A, np.eye(dim) + np.outer(x, x))
    assert np.allclose(retry_state.b, reward * x)

    assert np.array_equal(model.get_state(ActionType.PAYMENT_UPDATE).A, init_A_update)
    assert np.array_equal(model.get_state(ActionType.REMINDER).b, init_b_reminder)
    assert np.array_equal(model.get_state(ActionType.WAIT).A, init_A_wait)


# 10. Dimension mismatch rejection
def test_dimension_mismatch_rejection():
    """10. Verify dimension mismatch raises ValueError."""
    model = LinUCBValueModel(LinUCBConfig(dimension=20))

    with pytest.raises(ValueError, match="dimension mismatch"):
        model.predict_q1(np.ones(19), ActionType.RETRY)

    with pytest.raises(ValueError, match="dimension mismatch"):
        model.compute_exploration_bonus(np.ones(21), ActionType.WAIT)

    with pytest.raises(ValueError, match="dimension mismatch"):
        model.update(np.ones(5), ActionType.REMINDER, 10.0)


# 11. Zero vector behavior
def test_zero_vector_behavior():
    """11. Verify zero vector produces finite Q1=0 and Bonus=0."""
    dim = 20
    model = LinUCBValueModel(LinUCBConfig(dimension=dim))
    x_zero = np.zeros(dim)

    assert model.predict_q1(x_zero, ActionType.RETRY) == 0.0
    assert model.compute_exploration_bonus(x_zero, ActionType.RETRY) == 0.0


# 12. Near-singular matrix handling
def test_near_singular_matrix_stability():
    """12. Verify regularized A matrix maintains invertibility even with colinear updates."""
    dim = 3
    model = LinUCBValueModel(LinUCBConfig(dimension=dim, lambda_reg=0.1))
    x_colinear = np.array([1.0, 1.0, 1.0])

    for _ in range(50):
        model.update(x_colinear, ActionType.RETRY, 10.0)

    # Matrix has strong colinear rank-1 updates, but lambda * I regularizer preserves positive definiteness
    q1 = model.predict_q1(x_colinear, ActionType.RETRY)
    bonus = model.compute_exploration_bonus(x_colinear, ActionType.RETRY)
    assert np.isfinite(q1)
    assert np.isfinite(bonus)


# 13 & 14. Invalid alpha and lambda rejection
def test_invalid_hyperparameters_rejection():
    """13 & 14. Verify invalid alpha (<0) and lambda (<=0) are strictly rejected."""
    with pytest.raises(Exception):
        LinUCBConfig(alpha=-0.5)

    with pytest.raises(Exception):
        LinUCBConfig(lambda_reg=0.0)

    model = LinUCBValueModel()
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        model.compute_exploration_bonus(np.ones(20), ActionType.RETRY, custom_alpha=-1.0)


# 15. Non-finite input rejection
def test_non_finite_input_rejection():
    """15. Verify NaN and Inf in features or rewards are rejected."""
    dim = 20
    model = LinUCBValueModel(LinUCBConfig(dimension=dim))
    
    x_nan = np.ones(dim)
    x_nan[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        model.predict_q1(x_nan, ActionType.RETRY)

    x_inf = np.ones(dim)
    x_inf[2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        model.compute_exploration_bonus(x_inf, ActionType.WAIT)

    with pytest.raises(ValueError, match="finite"):
        model.update(np.ones(dim), ActionType.RETRY, np.nan)


# 16. Deterministic repeated calculation
def test_determinism_and_reproducibility():
    """16. Verify identical inputs produce bit-exact identical predictions."""
    model1 = LinUCBValueModel()
    model2 = LinUCBValueModel()
    x = np.linspace(0.1, 0.9, 20)

    assert model1.predict_q1(x, ActionType.RETRY) == model2.predict_q1(x, ActionType.RETRY)
    assert model1.compute_exploration_bonus(x, ActionType.RETRY) == model2.compute_exploration_bonus(x, ActionType.RETRY)


# 17. Model version propagation
def test_model_version_propagation():
    """17. Verify value_model_version string is exposed on the model."""
    config = LinUCBConfig(value_model_version="linucb_v5.3.1-custom")
    model = LinUCBValueModel(config)
    assert model.version == "linucb_v5.3.1-custom"


# 18. All four supported actions
def test_all_four_supported_actions_evaluated():
    """18. Verify all four supported actions can be predicted and updated."""
    model = LinUCBValueModel()
    x = np.ones(20) * 0.5

    for action in [ActionType.RETRY, ActionType.PAYMENT_UPDATE, ActionType.REMINDER, ActionType.WAIT]:
        q1 = model.predict_q1(x, action)
        bonus = model.compute_exploration_bonus(x, action)
        assert np.isfinite(q1)
        assert np.isfinite(bonus)


# 19 & 20. ESCALATE and STOP are NOT LinUCB actions
def test_escalate_and_stop_not_linucb_actions():
    """19 & 20. Verify ESCALATE and STOP raise errors when queried on LinUCB model."""
    model = LinUCBValueModel()
    x = np.ones(20)

    with pytest.raises(ValueError, match="not supported"):
        model.predict_q1(x, ActionType.ESCALATE)

    with pytest.raises(ValueError, match="not supported"):
        model.predict_q1(x, ActionType.STOP)

    with pytest.raises(ValueError, match="not supported"):
        model.compute_exploration_bonus(x, ActionType.ESCALATE)

    with pytest.raises(ValueError, match="not supported"):
        model.compute_exploration_bonus(x, ActionType.STOP)

    with pytest.raises(ValueError, match="not supported"):
        model.update(x, ActionType.ESCALATE, 100.0)

    with pytest.raises(ValueError, match="not supported"):
        model.update(x, ActionType.STOP, 0.0)
