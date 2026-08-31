"""Restricted-action heuristic model for ESCALATE score estimation."""
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile
from src.audit.schema import EstimationMethod

DEFAULT_HEURISTIC_VERSION: str = "heur_v5.0.0"


class EscalateHeuristicConfig(BaseModel):
    """Documented human-calibrated coefficients for restricted ESCALATE action.
    Score = amount_at_risk * escalation_factor + days_overdue * aging_factor + failed_attempts * failure_factor
    """
    heuristic_version: str = DEFAULT_HEURISTIC_VERSION
    escalation_factor: float = Field(default=0.5, ge=0.0, description="Weight for amount at risk")
    aging_factor: float = Field(default=2.0, ge=0.0, description="Weight for days overdue")
    failure_factor: float = Field(default=50.0, ge=0.0, description="Weight for previous failed attempts")


class EscalateHeuristicModel:
    """Computes documented human-calibrated heuristic score for ESCALATE.
    Explicitly tags estimation_method = 'heuristic'.
    """

    def __init__(self, config: EscalateHeuristicConfig | None = None):
        self.config = config or EscalateHeuristicConfig()
        self.version = self.config.heuristic_version

    def predict_heuristic_q(self, case: RecoveryCase, customer: CustomerProfile) -> float:
        """Compute documented heuristic score:
        H(x, ESCALATE) = amount * k_esc + days * k_aging + fails * k_fail
        """
        score = (
            float(case.amount_at_risk) * self.config.escalation_factor
            + float(case.days_overdue) * self.config.aging_factor
            + float(case.retry_attempt_count) * self.config.failure_factor
        )
        return float(score)
