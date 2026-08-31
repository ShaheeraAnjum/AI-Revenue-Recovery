"""Action-conditional transition probability model estimating P(s' | s, a)."""
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import CaseState, PaymentFailureCode
from src.context.schema import ContextFeatures, TOTAL_FEATURE_DIM

DEFAULT_TRANSITION_MODEL_VERSION: str = "trans_v5.0.0"


class RecoveryNextState(str, Enum):
    """Finite discrete state space representation for future lookahead sequences."""
    RECOVERED = "RECOVERED"
    STILL_AT_RISK = "STILL_AT_RISK"
    UNRECOVERABLE = "UNRECOVERABLE"


class TransitionDistribution(BaseModel):
    """Probability distribution over next states s' given current state and action."""
    action: ActionType
    probabilities: Dict[RecoveryNextState, float] = Field(
        ...,
        description="Probability distribution P(s' | s, a) summing to 1.0",
    )
    transition_model_version: str = DEFAULT_TRANSITION_MODEL_VERSION

    def get_probability(self, next_state: RecoveryNextState) -> float:
        """Get probability for a specific next state."""
        return self.probabilities.get(next_state, 0.0)


class TransitionModelConfig(BaseModel):
    """Configuration and calibration hyperparameters for action-conditional transition model."""
    transition_model_version: str = DEFAULT_TRANSITION_MODEL_VERSION
    dimension: int = Field(default=TOTAL_FEATURE_DIM, ge=1)
    tolerance: float = Field(default=1e-5, ge=0.0, description="Numerical tolerance for probability summation")


class ActionConditionalTransitionModel:
    """Estimates action-conditioned transition probabilities P(s' | s, a).
    Structurally enforces action-dependence so different actions yield distinct distributions.
    """

    def __init__(self, config: Optional[TransitionModelConfig] = None):
        self.config = config or TransitionModelConfig()
        self.version = self.config.transition_model_version
        self.dimension = self.config.dimension
        self.tolerance = self.config.tolerance

    def _validate_input(
        self,
        state: CaseState,
        action: ActionType,
        context: np.ndarray | List[float] | ContextFeatures,
    ) -> np.ndarray:
        """Validate inputs and return numeric feature vector x."""
        if isinstance(context, ContextFeatures):
            x = np.array(context.feature_vector, dtype=np.float64)
        else:
            x = np.asarray(context, dtype=np.float64)

        if x.ndim != 1 or len(x) != self.dimension:
            raise ValueError(
                f"Feature vector dimension mismatch: expected {self.dimension}, got {len(x)}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("Feature vector contains NaN or non-finite values")
        return x

    def _validate_distribution(
        self,
        probs: Dict[RecoveryNextState, float],
        action: ActionType,
    ) -> TransitionDistribution:
        """Ensure probabilities are non-negative, finite, and sum to 1.0 within tolerance."""
        total = 0.0
        validated: Dict[RecoveryNextState, float] = {}

        for state in RecoveryNextState:
            p = probs.get(state, 0.0)
            if not np.isfinite(p) or p < 0.0:
                raise ValueError(f"Invalid probability for state {state}: {p}")
            validated[state] = float(p)
            total += float(p)

        if abs(total - 1.0) > self.tolerance:
            raise ValueError(
                f"Transition probabilities for action {action} do not sum to 1.0 (sum={total})"
            )

        # Normalize minor floating-point residuals to ensure exact 1.0 sum
        for state in validated:
            validated[state] /= total

        return TransitionDistribution(
            action=action,
            probabilities=validated,
            transition_model_version=self.version,
        )

    def predict_transition(
        self,
        state: CaseState,
        action: ActionType,
        context: np.ndarray | List[float] | ContextFeatures,
    ) -> TransitionDistribution:
        """Predict transition distribution P(s' | s, a) conditioned explicitly on action a and context x."""
        x = self._validate_input(state, action, context)

        # Extract continuous context signals:
        # x[0] = amount_at_risk_norm, x[1] = days_overdue_norm, x[4] = previous_success_rate
        amount_norm = x[0]
        days_overdue_norm = x[1]
        success_rate = x[4] if len(x) > 4 else 0.5

        # Terminal action: STOP deterministically leads to UNRECOVERABLE
        if action == ActionType.STOP:
            probs = {
                RecoveryNextState.RECOVERED: 0.0,
                RecoveryNextState.STILL_AT_RISK: 0.0,
                RecoveryNextState.UNRECOVERABLE: 1.0,
            }
            return self._validate_distribution(probs, action)

        # Case already terminal
        if state in {CaseState.RESOLVED_RECOVERED, CaseState.RESOLVED_UNRECOVERABLE}:
            if state == CaseState.RESOLVED_RECOVERED:
                probs = {
                    RecoveryNextState.RECOVERED: 1.0,
                    RecoveryNextState.STILL_AT_RISK: 0.0,
                    RecoveryNextState.UNRECOVERABLE: 0.0,
                }
            else:
                probs = {
                    RecoveryNextState.RECOVERED: 0.0,
                    RecoveryNextState.STILL_AT_RISK: 0.0,
                    RecoveryNextState.UNRECOVERABLE: 1.0,
                }
            return self._validate_distribution(probs, action)

        # Action-specific calibrated conditional transition logic:
        if action == ActionType.RETRY:
            # RETRY success depends strongly on low days overdue and historical success rate
            p_rec = 0.40 * (1.0 - 0.5 * days_overdue_norm) + 0.35 * success_rate
            p_rec = max(0.05, min(0.85, p_rec))
            p_unrec = 0.10 + 0.30 * days_overdue_norm
            p_unrec = max(0.05, min(0.60, p_unrec))
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            probs = {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        elif action == ActionType.PAYMENT_UPDATE:
            # PAYMENT_UPDATE has high long-term recovery for expired cards and high customer engagement
            p_rec = 0.50 + 0.30 * success_rate - 0.20 * days_overdue_norm
            p_rec = max(0.10, min(0.90, p_rec))
            p_unrec = 0.05 + 0.25 * days_overdue_norm
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            probs = {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        elif action == ActionType.REMINDER:
            # REMINDER nudges customer to pay; moderate immediate recovery, mostly remains STILL_AT_RISK
            p_rec = 0.25 * (1.0 - 0.3 * days_overdue_norm) + 0.20 * success_rate
            p_rec = max(0.05, min(0.60, p_rec))
            p_unrec = 0.05 + 0.20 * days_overdue_norm
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            probs = {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        elif action == ActionType.WAIT:
            # WAIT has low immediate recovery, high probability of staying STILL_AT_RISK, aging risk
            p_rec = 0.08 * (1.0 - 0.7 * days_overdue_norm)
            p_rec = max(0.01, min(0.20, p_rec))
            p_unrec = 0.15 * days_overdue_norm + 0.05
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            probs = {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        elif action == ActionType.ESCALATE:
            # ESCALATE: human intervention, strong resolution probability on high-value/overdue cases
            p_rec = 0.60 * (1.0 - 0.2 * days_overdue_norm) + 0.20 * amount_norm
            p_rec = max(0.15, min(0.92, p_rec))
            p_unrec = 0.10 + 0.15 * days_overdue_norm
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            probs = {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        else:
            raise ValueError(f"Unknown action {action} in transition model")

        return self._validate_distribution(probs, action)
