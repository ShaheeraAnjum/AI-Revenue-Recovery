"""Unit tests for continuous context builder and feature extraction."""
import numpy as np
import pytest
from src.domain.case import (
    CustomerProfile,
    RecoveryCase,
    PaymentFailureCode,
    PaymentMethodType,
)
from src.context.builder import ContextBuilder
from src.context.schema import FEATURE_NAMES, NUM_FEATURES


def test_context_builder_dimensions():
    """Verify continuous feature vector x has exact required dimensions (8 continuous features)."""
    customer = CustomerProfile(
        customer_id="CUST-100",
        customer_value=5000.0,
        subscription_age_days=90,
        previous_success_rate=0.75,
        previous_contact_count=1,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
    )
    case = RecoveryCase(
        case_id="CASE-100",
        customer_id="CUST-100",
        amount_at_risk=4280.0,
        failure_code=PaymentFailureCode.CARD_EXPIRED,
        days_overdue=4,
    )

    builder = ContextBuilder()
    context = builder.build_context(case=case, customer=customer)

    assert len(context.feature_vector) == NUM_FEATURES
    assert len(FEATURE_NAMES) == 8

    vec_np = builder.to_numpy(context)
    assert isinstance(vec_np, np.ndarray)
    assert vec_np.shape == (8,)
    assert vec_np.dtype == np.float64

    assert vec_np[0] == 4280.0
    assert vec_np[2] == 4.0
    assert vec_np[3] == 5000.0
    assert vec_np[4] == 90.0
    assert vec_np[5] == 0.75
    assert vec_np[6] == 1.0


def test_context_features_dict_mapping():
    """Verify dictionary mapping for auditing and logging."""
    customer = CustomerProfile(
        customer_id="CUST-200",
        customer_value=1200.0,
        subscription_age_days=30,
        previous_success_rate=0.5,
        previous_contact_count=0,
        payment_method_type=PaymentMethodType.UPI,
    )
    case = RecoveryCase(
        case_id="CASE-200",
        customer_id="CUST-200",
        amount_at_risk=150.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=1,
    )

    builder = ContextBuilder(schema_version="v1.0.0")
    context = builder.build_context(case, customer)
    d = context.to_dict()

    for name in FEATURE_NAMES:
        assert name in d
    assert d["amount_at_risk"] == 150.0
    assert d["days_overdue"] == 1.0
    assert context.feature_schema_version == "v1.0.0"
