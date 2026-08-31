"""Configuration and versioning for LinUCB contextual bandit value models."""
from pydantic import BaseModel, Field
from src.context.schema import TOTAL_FEATURE_DIM

DEFAULT_VALUE_MODEL_VERSION: str = "linucb_v5.0.0"


class LinUCBConfig(BaseModel):
    """Configuration hyperparameters for contextual LinUCB model."""
    value_model_version: str = DEFAULT_VALUE_MODEL_VERSION
    dimension: int = Field(
        default=TOTAL_FEATURE_DIM,
        ge=1,
        description="Feature vector dimension d (must match FeatureSchema)",
    )
    alpha: float = Field(
        default=1.0,
        ge=0.0,
        description="Exploration parameter alpha in B(x, a) = alpha * sqrt(x^T A_a^-1 x)",
    )
    lambda_reg: float = Field(
        default=1.0,
        gt=0.0,
        description="L2 regularization parameter lambda for A_a = lambda * I_d initialization",
    )
