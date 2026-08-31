"""Continuous feature schema, scaling bounds, and one-hot encoding definitions for LinUCB."""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.domain.case import PaymentFailureCode, PaymentMethodType

# Deterministic enumeration order for one-hot encodings
ORDERED_FAILURE_CODES: List[PaymentFailureCode] = list(PaymentFailureCode)
ORDERED_PAYMENT_METHODS: List[PaymentMethodType] = list(PaymentMethodType)

# Continuous numeric features (6 features)
NUMERIC_FEATURE_NAMES: List[str] = [
    "amount_at_risk_norm",
    "days_overdue_norm",
    "customer_value_norm",
    "subscription_age_norm",
    "previous_success_rate",
    "previous_contact_count_norm",
]

# Categorical one-hot features (9 failure codes + 5 payment methods = 14 features)
CATEGORICAL_FEATURE_NAMES: List[str] = (
    [f"fail_code_{code.value}" for code in ORDERED_FAILURE_CODES]
    + [f"pay_method_{method.value}" for method in ORDERED_PAYMENT_METHODS]
)

# Total feature vector names in canonical deterministic order (6 + 14 = 20 dimensions)
CANONICAL_FEATURE_NAMES: List[str] = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
TOTAL_FEATURE_DIM: int = len(CANONICAL_FEATURE_NAMES)
DEFAULT_FEATURE_SCHEMA_VERSION: str = "v1.1.0"


class FeatureScaleConfig(BaseModel):
    """Versioned deterministic normalization bounds and scale divisors for continuous inputs."""
    amount_at_risk_scale: float = Field(default=10000.0, description="Scale divisor for amount at risk")
    days_overdue_scale: float = Field(default=90.0, description="Scale divisor for days overdue (clamped [0, 1])")
    customer_value_scale: float = Field(default=50000.0, description="Scale divisor for customer lifetime value")
    subscription_age_scale: float = Field(default=3650.0, description="Scale divisor for subscription age in days (10y)")
    contact_count_scale: float = Field(default=20.0, description="Scale divisor for prior contact attempts")
    clip_bounds: bool = Field(default=True, description="Whether to clip normalized numeric features to [0.0, 1.0]")


class FeatureSchemaVersion(BaseModel):
    """Schema metadata and transformation parameters for full offline reproducibility."""
    version: str = DEFAULT_FEATURE_SCHEMA_VERSION
    feature_names: List[str] = Field(default_factory=lambda: list(CANONICAL_FEATURE_NAMES))
    num_features: int = TOTAL_FEATURE_DIM
    scale_config: FeatureScaleConfig = Field(default_factory=FeatureScaleConfig)


class ContextFeatures(BaseModel):
    """Normalized continuous context vector x in R^d and raw context dictionary."""
    case_id: str
    customer_id: str
    raw_features: Dict[str, Any] = Field(..., description="Raw input metrics before transformation")
    normalized_numeric_features: Dict[str, float] = Field(..., description="Normalized continuous numeric features")
    categorical_encodings: Dict[str, float] = Field(..., description="One-hot categorical encodings")
    feature_vector: List[float] = Field(..., description="Deterministic continuous feature vector x in R^d")
    feature_schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary for logging and audit serialization."""
        return {
            **self.normalized_numeric_features,
            **self.categorical_encodings,
        }
