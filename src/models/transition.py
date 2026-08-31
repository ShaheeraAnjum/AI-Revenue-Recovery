"""Action-conditional transition probability model estimating P(s' | s, a, x)."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import CaseState
from src.context.schema import ContextFeatures, TOTAL_FEATURE_DIM, CANONICAL_FEATURE_NAMES

DEFAULT_TRANSITION_MODEL_VERSION: str = "trans_v5.0.0"

# Canonical feature name index mapping to eliminate magic array indexing
FEATURE_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CANONICAL_FEATURE_NAMES)}


class RecoveryNextState(str, Enum):
    """Finite discrete state space representation for future lookahead sequences.
    Mapping from CaseState:
    - DETECTED / ACTIVE -> transitions to RECOVERED, STILL_AT_RISK, or UNRECOVERABLE.
    - RESOLVED_RECOVERED -> absorbing terminal state (P(RECOVERED) = 1.0).
    - RESOLVED_UNRECOVERABLE -> absorbing terminal state (P(UNRECOVERABLE) = 1.0).
    - STOP action -> deterministic transition to UNRECOVERABLE (P(UNRECOVERABLE) = 1.0).
    """
    RECOVERED = "RECOVERED"
    STILL_AT_RISK = "STILL_AT_RISK"
    UNRECOVERABLE = "UNRECOVERABLE"


class TransitionEstimationMethod(str, Enum):
    """Explicit estimation provenance tagging for transition probabilities."""
    CALIBRATED_PRIOR = "calibrated_prior"
    HEURISTIC = "heuristic"
    LEARNED = "learned"


class TransitionDistribution(BaseModel):
    """Audit-ready transition probability distribution P(s' | s, a, x)."""
    current_state: CaseState
    action: ActionType
    probabilities: Dict[RecoveryNextState, float] = Field(
        ...,
        description="Probability distribution P(s' | s, a, x) summing to 1.0",
    )
    transition_model_version: str = DEFAULT_TRANSITION_MODEL_VERSION
    estimation_method: TransitionEstimationMethod = TransitionEstimationMethod.CALIBRATED_PRIOR

    def get_probability(self, next_state: RecoveryNextState) -> float:
        """Get probability for a specific next state."""
        return self.probabilities.get(next_state, 0.0)


class BaseTransitionEstimator(ABC):
    """Abstract interface for action-conditional transition estimation."""

    @abstractmethod
    def estimate(
        self,
        state: CaseState,
        action: ActionType,
        named_features: Dict[str, float],
    ) -> Dict[RecoveryNextState, float]:
        """Estimate next-state probability distribution given current state, action, and named features."""
        pass

    @property
    @abstractmethod
    def estimation_method(self) -> TransitionEstimationMethod:
        """Return explicit provenance category for this estimator."""
        pass


class CalibratedPriorTransitionEstimator(BaseTransitionEstimator):
    """Documented prototype calibration-prior estimator.
    NOTE: These probabilities represent domain-calibrated initialization priors for the prototype
    system and are explicitly NOT claims of learned causal evidence from observational data.
    """

    @property
    def estimation_method(self) -> TransitionEstimationMethod:
        return TransitionEstimationMethod.CALIBRATED_PRIOR

    def estimate(
        self,
        state: CaseState,
        action: ActionType,
        named_features: Dict[str, float],
    ) -> Dict[RecoveryNextState, float]:
        """Estimate transition distribution using documented feature semantics."""
        # Named feature lookups (no raw magic indices)
        amount_norm = named_features.get("amount_at_risk_norm", 0.5)
        days_overdue_norm = named_features.get("days_overdue_norm", 0.0)
        success_rate = named_features.get("previous_success_rate", 0.5)
        contact_norm = named_features.get("previous_contact_count_norm", 0.0)

        # Terminal action: STOP deterministically leads to UNRECOVERABLE
        if action == ActionType.STOP:
            return {
                RecoveryNextState.RECOVERED: 0.0,
                RecoveryNextState.STILL_AT_RISK: 0.0,
                RecoveryNextState.UNRECOVERABLE: 1.0,
            }

        # Terminal states
        if state == CaseState.RESOLVED_RECOVERED:
            return {
                RecoveryNextState.RECOVERED: 1.0,
                RecoveryNextState.STILL_AT_RISK: 0.0,
                RecoveryNextState.UNRECOVERABLE: 0.0,
            }
        if state == CaseState.RESOLVED_UNRECOVERABLE:
            return {
                RecoveryNextState.RECOVERED: 0.0,
                RecoveryNextState.STILL_AT_RISK: 0.0,
                RecoveryNextState.UNRECOVERABLE: 1.0,
            }

        # Action-specific calibrated conditional transition priors:
        if action == ActionType.RETRY:
            p_rec = 0.40 * (1.0 - 0.5 * days_overdue_norm) + 0.35 * success_rate
            p_rec = max(0.05, min(0.85, p_rec))
            p_unrec = 0.10 + 0.30 * days_overdue_norm
            p_unrec = max(0.05, min(0.60, p_unrec))
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            return {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        elif action == ActionType.PAYMENT_UPDATE:
            p_rec = 0.50 + 0.30 * success_rate - 0.20 * days_overdue_norm
            p_rec = max(0.10, min(0.90, p_rec))
            p_unrec = 0.05 + 0.25 * days_overdue_norm
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            return {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        elif action == ActionType.REMINDER:
            p_rec = 0.25 * (1.0 - 0.3 * days_overdue_norm) + 0.20 * success_rate - 0.10 * contact_norm
            p_rec = max(0.05, min(0.60, p_rec))
            p_unrec = 0.05 + 0.20 * days_overdue_norm
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            return {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        elif action == ActionType.WAIT:
            p_rec = 0.08 * (1.0 - 0.7 * days_overdue_norm)
            p_rec = max(0.01, min(0.20, p_rec))
            p_unrec = 0.15 * days_overdue_norm + 0.05
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            return {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        elif action == ActionType.ESCALATE:
            p_rec = 0.60 * (1.0 - 0.2 * days_overdue_norm) + 0.20 * amount_norm
            p_rec = max(0.15, min(0.92, p_rec))
            p_unrec = 0.10 + 0.15 * days_overdue_norm
            p_risk = max(0.0, 1.0 - p_rec - p_unrec)
            return {
                RecoveryNextState.RECOVERED: p_rec,
                RecoveryNextState.STILL_AT_RISK: p_risk,
                RecoveryNextState.UNRECOVERABLE: p_unrec,
            }

        raise ValueError(f"Unknown action {action} in transition estimator")


class TransitionModelConfig(BaseModel):
    """Configuration and calibration hyperparameters for action-conditional transition model."""
    transition_model_version: str = DEFAULT_TRANSITION_MODEL_VERSION
    dimension: int = Field(default=TOTAL_FEATURE_DIM, ge=1)
    tolerance: float = Field(default=1e-5, ge=0.0, description="Numerical tolerance for probability summation")


class ActionConditionalTransitionModel:
    """Estimates action-conditioned transition probabilities P(s' | s, a, x).
    Validates input vectors and enforces probability distribution invariants.
    """

    def __init__(
        self,
        config: Optional[TransitionModelConfig] = None,
        estimator: Optional[BaseTransitionEstimator] = None,
    ):
        self.config = config or TransitionModelConfig()
        self.version = self.config.transition_model_version
        self.dimension = self.config.dimension
        self.tolerance = self.config.tolerance
        self.estimator = estimator or CalibratedPriorTransitionEstimator()

    def _extract_named_features(
        self,
        context: np.ndarray | List[float] | ContextFeatures,
    ) -> Dict[str, float]:
        """Extract named features from context vector or ContextFeatures object without magic indices."""
        if isinstance(context, ContextFeatures):
            return context.to_dict()

        x = np.asarray(context, dtype=np.float64)
        if x.ndim != 1 or len(x) != self.dimension:
            raise ValueError(
                f"Feature vector dimension mismatch: expected {self.dimension}, got {len(x)}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("Feature vector contains NaN or non-finite values")

        named: Dict[str, float] = {}
        for name, idx in FEATURE_IDX.items():
            if idx < len(x):
                named[name] = float(x[idx])
        return named

    def _validate_distribution(
        self,
        raw_probs: Dict[RecoveryNextState, float],
        state: CaseState,
        action: ActionType,
    ) -> TransitionDistribution:
        """Ensure probabilities are non-negative, finite, and sum to 1.0 within tolerance."""
        total = 0.0
        validated: Dict[RecoveryNextState, float] = {}

        for s in RecoveryNextState:
            p = raw_probs.get(s, 0.0)
            if not np.isfinite(p) or p < 0.0:
                raise ValueError(f"Invalid probability for state {s}: {p}")
            validated[s] = float(p)
            total += float(p)

        if abs(total - 1.0) > self.tolerance:
            raise ValueError(
                f"Transition probabilities for action {action} do not sum to 1.0 (sum={total})"
            )

        # Normalize minor floating-point residuals to ensure exact 1.0 sum
        for s in validated:
            validated[s] /= total

        return TransitionDistribution(
            current_state=state,
            action=action,
            probabilities=validated,
            transition_model_version=self.version,
            estimation_method=self.estimator.estimation_method,
        )

    def predict_transition(
        self,
        state: CaseState,
        action: ActionType,
        context: np.ndarray | List[float] | ContextFeatures,
    ) -> TransitionDistribution:
        """Predict transition distribution P(s' | s, a, x) conditioned explicitly on action a and context x."""
        named_features = self._extract_named_features(context)
        raw_probs = self.estimator.estimate(state, action, named_features)
        return self._validate_distribution(raw_probs, state, action)
