"""Context builder extracting deterministic, normalized continuous feature vector x in R^d for LinUCB."""
import numpy as np
from typing import List, Dict, Any
from src.domain.case import RecoveryCase, CustomerProfile, PaymentFailureCode, PaymentMethodType
from src.context.schema import (
    ContextFeatures,
    FeatureScaleConfig,
    FeatureSchemaVersion,
    DEFAULT_FEATURE_SCHEMA_VERSION,
    CANONICAL_FEATURE_NAMES,
    ORDERED_FAILURE_CODES,
    ORDERED_PAYMENT_METHODS,
    TOTAL_FEATURE_DIM,
)


class ContextBuilder:
    """Constructs the deterministic continuous feature vector x in R^d from customer and case context."""

    def __init__(
        self,
        schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION,
        scale_config: FeatureScaleConfig | None = None,
    ):
        self.schema_version = schema_version
        self.scale_config = scale_config or FeatureScaleConfig()
        self.schema = FeatureSchemaVersion(
            version=self.schema_version,
            feature_names=list(CANONICAL_FEATURE_NAMES),
            num_features=TOTAL_FEATURE_DIM,
            scale_config=self.scale_config,
        )

    def _normalize(self, value: float, scale: float, clip: bool = True) -> float:
        """Deterministically scale and optionally clamp numeric value to [0.0, 1.0]."""
        if scale <= 0.0:
            return 0.0
        norm = float(value) / float(scale)
        if clip:
            norm = max(0.0, min(1.0, norm))
        return float(norm)

    def build_context(self, case: RecoveryCase, customer: CustomerProfile) -> ContextFeatures:
        """Extract and transform raw context into normalized vector x in R^20 with one-hot categorical encoding."""
        # 1. Raw features record
        raw_dict: Dict[str, Any] = {
            "amount_at_risk": float(case.amount_at_risk),
            "failure_code": case.failure_code.value,
            "days_overdue": int(case.days_overdue),
            "customer_value": float(customer.customer_value),
            "subscription_age_days": int(customer.subscription_age_days),
            "previous_success_rate": float(customer.previous_success_rate),
            "previous_contact_count": int(customer.previous_contact_count),
            "payment_method_type": customer.payment_method_type.value,
            "days_waiting": int(case.days_waiting),
            "active_recovery_cases": int(customer.active_recovery_cases),
        }

        # 2. Normalized continuous numeric features (6 features)
        clip = self.scale_config.clip_bounds
        amount_norm = self._normalize(case.amount_at_risk, self.scale_config.amount_at_risk_scale, clip)
        days_overdue_norm = self._normalize(case.days_overdue, self.scale_config.days_overdue_scale, clip)
        cust_val_norm = self._normalize(customer.customer_value, self.scale_config.customer_value_scale, clip)
        sub_age_norm = self._normalize(customer.subscription_age_days, self.scale_config.subscription_age_scale, clip)
        success_rate = max(0.0, min(1.0, float(customer.previous_success_rate)))
        contact_count_norm = self._normalize(customer.previous_contact_count, self.scale_config.contact_count_scale, clip)

        norm_numeric: Dict[str, float] = {
            "amount_at_risk_norm": amount_norm,
            "days_overdue_norm": days_overdue_norm,
            "customer_value_norm": cust_val_norm,
            "subscription_age_norm": sub_age_norm,
            "previous_success_rate": success_rate,
            "previous_contact_count_norm": contact_count_norm,
        }

        # 3. Categorical one-hot encodings (9 + 5 = 14 features)
        cat_encodings: Dict[str, float] = {}
        for code in ORDERED_FAILURE_CODES:
            cat_encodings[f"fail_code_{code.value}"] = 1.0 if case.failure_code == code else 0.0

        for method in ORDERED_PAYMENT_METHODS:
            cat_encodings[f"pay_method_{method.value}"] = 1.0 if customer.payment_method_type == method else 0.0

        # 4. Construct canonical ordered feature vector x in R^d
        vector: List[float] = [
            amount_norm,
            days_overdue_norm,
            cust_val_norm,
            sub_age_norm,
            success_rate,
            contact_count_norm,
        ]
        for code in ORDERED_FAILURE_CODES:
            vector.append(cat_encodings[f"fail_code_{code.value}"])
        for method in ORDERED_PAYMENT_METHODS:
            vector.append(cat_encodings[f"pay_method_{method.value}"])

        assert len(vector) == TOTAL_FEATURE_DIM, f"Vector dimension {len(vector)} != {TOTAL_FEATURE_DIM}"

        return ContextFeatures(
            case_id=case.case_id,
            customer_id=customer.customer_id,
            raw_features=raw_dict,
            normalized_numeric_features=norm_numeric,
            categorical_encodings=cat_encodings,
            feature_vector=vector,
            feature_schema_version=self.schema_version,
        )

    def to_numpy(self, context: ContextFeatures) -> np.ndarray:
        """Return NumPy array of shape (d,) representing feature vector x."""
        return np.array(context.feature_vector, dtype=np.float64)
