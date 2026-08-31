"""Context builder to extract continuous feature vector x for LinUCB."""
import numpy as np
from typing import List
from src.domain.case import RecoveryCase, CustomerProfile, PaymentFailureCode, PaymentMethodType
from src.context.schema import ContextFeatures, DEFAULT_FEATURE_SCHEMA_VERSION, FEATURE_NAMES

FAILURE_CODE_MAP = {code: float(i) for i, code in enumerate(PaymentFailureCode)}
PAYMENT_METHOD_MAP = {method: float(i) for i, method in enumerate(PaymentMethodType)}


class ContextBuilder:
    """Constructs the continuous feature vector x from customer and case context."""

    def __init__(self, schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION):
        self.schema_version = schema_version

    def build_context(self, case: RecoveryCase, customer: CustomerProfile) -> ContextFeatures:
        """Extract continuous feature vector x = [
            amount_at_risk,
            failure_code,
            days_overdue,
            customer_value,
            subscription_age,
            previous_success_rate,
            previous_contact_count,
            payment_method_type
        ]"""
        fail_idx = FAILURE_CODE_MAP.get(case.failure_code, 0.0)
        pay_idx = PAYMENT_METHOD_MAP.get(customer.payment_method_type, 0.0)

        vec: List[float] = [
            float(case.amount_at_risk),
            float(fail_idx),
            float(case.days_overdue),
            float(customer.customer_value),
            float(customer.subscription_age_days),
            float(customer.previous_success_rate),
            float(customer.previous_contact_count),
            float(pay_idx),
        ]

        return ContextFeatures(
            case_id=case.case_id,
            customer_id=customer.customer_id,
            amount_at_risk=float(case.amount_at_risk),
            failure_code_idx=float(fail_idx),
            days_overdue=float(case.days_overdue),
            customer_value=float(customer.customer_value),
            subscription_age=float(customer.subscription_age_days),
            previous_success_rate=float(customer.previous_success_rate),
            previous_contact_count=float(customer.previous_contact_count),
            payment_method_idx=float(pay_idx),
            feature_vector=vec,
            feature_schema_version=self.schema_version,
        )

    def to_numpy(self, context: ContextFeatures) -> np.ndarray:
        """Return NumPy array of shape (d,) representing feature vector x."""
        return np.array(context.feature_vector, dtype=np.float64)
