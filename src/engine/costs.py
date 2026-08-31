"""Action cost models and dynamic WAIT cost calculator."""
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase

DEFAULT_COST_CONFIG_VERSION: str = "cost_v5.0.0"


class CostConfig(BaseModel):
    """Versioned cost configuration parameters."""
    cost_config_version: str = DEFAULT_COST_CONFIG_VERSION
    
    # WAIT dynamic cost rates (holding cost and delay risk rate per day)
    r_hold: float = Field(default=10.0, ge=0.0, description="Configurable holding cost rate per day of waiting")
    r_delay: float = Field(default=15.0, ge=0.0, description="Configurable delay risk rate per day overdue")
    
    # Fixed operational / channel costs
    retry_cost: float = Field(default=2.0, ge=0.0, description="Gateway processing fee per automated retry")
    payment_update_cost: float = Field(default=1.0, ge=0.0, description="Secure portal link delivery cost")
    reminder_cost: float = Field(default=0.5, ge=0.0, description="Outbound SMS/Email reminder delivery cost")
    escalation_cost: float = Field(default=25.0, ge=0.0, description="Human agent handling cost per escalation")
    stop_cost: float = Field(default=0.0, ge=0.0, description="Cost of terminating recovery (0.0)")


class ActionCostCalculator:
    """Calculates exact action execution and dynamic delay costs C(s, a)."""

    def __init__(self, config: CostConfig | None = None):
        self.config = config or CostConfig()

    def calculate_cost(self, action: ActionType, case: RecoveryCase) -> float:
        """Compute cost C(s, a).
        For WAIT: C_wait = r_hold * days_waiting + r_delay * days_overdue
        """
        if action == ActionType.WAIT:
            c_hold = self.config.r_hold * float(case.days_waiting)
            c_delay = self.config.r_delay * float(case.days_overdue)
            return float(c_hold + c_delay)
        elif action == ActionType.RETRY:
            return float(self.config.retry_cost)
        elif action == ActionType.PAYMENT_UPDATE:
            return float(self.config.payment_update_cost)
        elif action == ActionType.REMINDER:
            return float(self.config.reminder_cost)
        elif action == ActionType.ESCALATE:
            return float(self.config.escalation_cost)
        elif action == ActionType.STOP:
            return float(self.config.stop_cost)
        else:
            raise ValueError(f"Unknown action {action} in cost calculation")
