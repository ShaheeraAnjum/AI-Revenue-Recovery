"""Action-specific contextual LinUCB value model for supported recovery actions."""
import numpy as np
from typing import Dict, List, Optional
from src.domain.actions import ActionType, SUPPORTED_ACTIONS, is_supported_action
from src.context.schema import ContextFeatures, TOTAL_FEATURE_DIM
from src.models.config import LinUCBConfig, DEFAULT_VALUE_MODEL_VERSION


class ActionLinUCBState:
    """Independent LinUCB parameter state for a single supported action."""

    def __init__(self, action: ActionType, dimension: int, lambda_reg: float):
        if not is_supported_action(action):
            raise ValueError(f"Action {action} is not supported by LinUCB contextual model")
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {dimension}")
        if lambda_reg <= 0.0:
            raise ValueError(f"lambda_reg must be > 0, got {lambda_reg}")

        self.action = action
        self.dimension = dimension
        self.lambda_reg = float(lambda_reg)
        
        # A_a = lambda * I_d
        self.A: np.ndarray = self.lambda_reg * np.eye(dimension, dtype=np.float64)
        # b_a = 0_d
        self.b: np.ndarray = np.zeros(dimension, dtype=np.float64)
        self.update_count: int = 0

    @property
    def theta(self) -> np.ndarray:
        """Compute point estimate theta_a = A_a^-1 b_a using stable linear solver."""
        return np.linalg.solve(self.A, self.b)

    def predict_q1(self, x: np.ndarray) -> float:
        """Predict base contextual value Q1(x, a) = x^T theta_a."""
        theta = self.theta
        return float(np.dot(x, theta))

    def compute_uncertainty(self, x: np.ndarray) -> float:
        """Compute normalized uncertainty term sqrt(x^T A_a^-1 x)."""
        # Solve A_a y = x -> y = A_a^-1 x
        y = np.linalg.solve(self.A, x)
        variance = float(np.dot(x, y))
        # Numerical safeguard against tiny negative precision drift
        variance = max(0.0, variance)
        return float(np.sqrt(variance))

    def update(self, x: np.ndarray, reward: float) -> None:
        """Perform rank-1 outer product update:
        A_a <- A_a + x x^T
        b_a <- b_a + r x
        """
        self.A += np.outer(x, x)
        self.b += float(reward) * x
        self.update_count += 1


class LinUCBValueModel:
    """Contextual LinUCB value model managing independent parameter states for all supported actions."""

    def __init__(self, config: Optional[LinUCBConfig] = None):
        self.config = config or LinUCBConfig()
        self.dimension = self.config.dimension
        self.alpha = self.config.alpha
        self.lambda_reg = self.config.lambda_reg
        self.version = self.config.value_model_version

        # Initialize independent state for each supported action
        self.states: Dict[ActionType, ActionLinUCBState] = {
            action: ActionLinUCBState(
                action=action,
                dimension=self.dimension,
                lambda_reg=self.lambda_reg,
            )
            for action in SUPPORTED_ACTIONS
        }

    def _validate_vector(self, x: np.ndarray) -> np.ndarray:
        """Validate vector dimensions and numeric finiteness."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"Feature vector must be 1-dimensional, got shape {x.shape}")
        if len(x) != self.dimension:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.dimension}, got {len(x)}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("Feature vector contains NaN or non-finite values")
        return x

    def predict_q1(self, x: np.ndarray | List[float] | ContextFeatures, action: ActionType) -> float:
        """Predict base contextual value Q1(x, a) = x^T theta_a for supported action a."""
        if not is_supported_action(action):
            raise ValueError(f"Action {action} is not supported by LinUCB contextual model")

        if isinstance(x, ContextFeatures):
            x_vec = np.array(x.feature_vector, dtype=np.float64)
        else:
            x_vec = np.asarray(x, dtype=np.float64)

        x_vec = self._validate_vector(x_vec)
        return self.states[action].predict_q1(x_vec)

    def compute_exploration_bonus(
        self,
        x: np.ndarray | List[float] | ContextFeatures,
        action: ActionType,
        custom_alpha: Optional[float] = None,
    ) -> float:
        """Compute exploration bonus B(x, a) = alpha * sqrt(x^T A_a^-1 x).
        MANDATORY: Uses exact square root of the quadratic uncertainty form.
        """
        if not is_supported_action(action):
            raise ValueError(f"Action {action} is not supported by LinUCB contextual model")

        if isinstance(x, ContextFeatures):
            x_vec = np.array(x.feature_vector, dtype=np.float64)
        else:
            x_vec = np.asarray(x, dtype=np.float64)

        x_vec = self._validate_vector(x_vec)
        alpha = self.alpha if custom_alpha is None else custom_alpha
        if alpha < 0.0:
            raise ValueError(f"Exploration parameter alpha must be >= 0, got {alpha}")

        uncertainty = self.states[action].compute_uncertainty(x_vec)
        return float(alpha * uncertainty)

    def predict_q_explore(
        self,
        x: np.ndarray | List[float] | ContextFeatures,
        action: ActionType,
        custom_alpha: Optional[float] = None,
    ) -> float:
        """Compute full LinUCB upper confidence bound Q_explore(x, a) = Q1(x, a) + B(x, a)."""
        q1 = self.predict_q1(x, action)
        bonus = self.compute_exploration_bonus(x, action, custom_alpha)
        return float(q1 + bonus)

    def update(
        self,
        x: np.ndarray | List[float] | ContextFeatures,
        action: ActionType,
        reward: float,
    ) -> None:
        """Update parameter matrices for the selected supported action only:
        A_a <- A_a + x x^T
        b_a <- b_a + r x
        """
        if not is_supported_action(action):
            raise ValueError(f"Action {action} is not supported by LinUCB contextual model")
        if not np.isfinite(reward):
            raise ValueError(f"Reward must be finite, got {reward}")

        if isinstance(x, ContextFeatures):
            x_vec = np.array(x.feature_vector, dtype=np.float64)
        else:
            x_vec = np.asarray(x, dtype=np.float64)

        x_vec = self._validate_vector(x_vec)
        self.states[action].update(x_vec, reward)

    def get_state(self, action: ActionType) -> ActionLinUCBState:
        """Retrieve parameter state for a specific supported action."""
        if not is_supported_action(action):
            raise ValueError(f"Action {action} is not supported by LinUCB contextual model")
        return self.states[action]
