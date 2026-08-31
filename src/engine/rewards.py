"""Action reward model decoupled from transition probability calculations."""
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile, PaymentFailureCode

DEFAULT_REWARD_CONFIG_VERSION: str = "reward_v5.0.0"


class RewardConfig(BaseModel):
    """Versioned configuration parameters for action-specific reward estimation."""
    reward_config_version: str = DEFAULT_REWARD_CONFIG_VERSION
    
    # Base recovery yield factors per action type (gross economic value before cost)
    retry_yield_factor: float = Field(default=0.65, ge=0.0, le=1.0, description="Gross yield factor for successful retry")
    payment_update_yield_factor: float = Field(default=0.70, ge=0.0, le=1.0, description="Gross yield factor for payment update link")
    reminder_yield_factor: float = Field(default=0.45, ge=0.0, le=1.0, description="Gross yield factor for customer reminder")
    escalation_yield_factor: float = Field(default=0.80, ge=0.0, le=1.0, description="Gross yield factor for human agent escalation")
    wait_yield_factor: float = Field(default=0.10, ge=0.0, le=1.0, description="Passive natural cure yield factor during wait")
    stop_yield_factor: float = Field(default=0.0, ge=0.0, le=0.0, description="Gross yield for terminal STOP action (0.0)")


class ActionRewardCalculator:
    """Calculates action-dependent expected gross economic reward R(s, a).
    Explicitly decoupled from the transition probability estimator.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.version = self.config.reward_config_version

    def calculate_reward(
        self,
        action: ActionType,
        case: RecoveryCase,
        customer: CustomerProfile,
    ) -> float:
        """Compute action-dependent expected reward R(s, a).
        For STOP: R(STOP) = 0.0
        For active recovery: R(s, a) = amount_at_risk * yield_factor(a) * customer_retention_multiplier
        """
        if action == ActionType.STOP:
            return 0.0

        amount = float(case.amount_at_risk)
        success_mult = 0.5 + 0.5 * float(customer.previous_success_rate)

        if action == ActionType.RETRY:
            factor = self.config.retry_yield_factor
        elif action == ActionType.PAYMENT_UPDATE:
            factor = self.config.payment_update_yield_factor
        elif action == ActionType.REMINDER:
            factor = self.config.reminder_yield_factor
        elif action == ActionType.ESCALATE:
            factor = self.config.escalation_yield_factor
        elif action == ActionType.WAIT:
            factor = self.config.wait_yield_factor
        else:
            raise ValueError(f"Unknown action {action} in reward calculation")

        return float(amount * factor * success_mult)
