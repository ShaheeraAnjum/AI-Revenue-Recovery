"""Continuous feature schema and definitions for LinUCB contextual bandit."""
from typing import List, Dict
from pydantic import BaseModel, Field

FEATURE_NAMES: List[str] = [
    "amount_at_risk",
    "failure_code_idx",
    "days_overdue",
    "customer_value",
    "subscription_age",
    "previous_success_rate",
    "previous_contact_count",
    "payment_method_idx",
]

NUM_FEATURES: int = len(FEATURE_NAMES)
DEFAULT_FEATURE_SCHEMA_VERSION: str = "v1.0.0"


class FeatureSchemaVersion(BaseModel):
    """Schema version metadata for continuous context."""
    version: str = DEFAULT_FEATURE_SCHEMA_VERSION
    feature_names: List[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    num_features: int = NUM_FEATURES


class ContextFeatures(BaseModel):
    """Continuous feature representations of customer and recovery case."""
    case_id: str
    customer_id: str
    amount_at_risk: float
    failure_code_idx: float
    days_overdue: float
    customer_value: float
    subscription_age: float
    previous_success_rate: float
    previous_contact_count: float
    payment_method_idx: float
    feature_vector: List[float] = Field(..., description="Continuous numeric vector x in R^d")
    feature_schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary."""
        return {
            "amount_at_risk": self.amount_at_risk,
            "failure_code_idx": self.failure_code_idx,
            "days_overdue": self.days_overdue,
            "customer_value": self.customer_value,
            "subscription_age": self.subscription_age,
            "previous_success_rate": self.previous_success_rate,
            "previous_contact_count": self.previous_contact_count,
            "payment_method_idx": self.payment_method_idx,
        }
