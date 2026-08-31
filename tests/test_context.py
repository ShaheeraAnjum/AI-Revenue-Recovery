"""Unit tests for continuous context builder, one-hot encoding, normalization, and reproducibility."""
import numpy as np
import pytest
from src.domain.case import (
    CustomerProfile,
    RecoveryCase,
    PaymentFailureCode,
    PaymentMethodType,
)
from src.context.builder import ContextBuilder
from src.context.schema import (
    CANONICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    CATEGORICAL_FEATURE_NAMES,
    ORDERED_FAILURE_CODES,
    ORDERED_PAYMENT_METHODS,
    TOTAL_FEATURE_DIM,
    FeatureScaleConfig,
)


def test_feature_schema_dimension_and_ordering():
    """Verify feature vector dimension matches schema declared dimension (20 total dimensions)."""
    assert TOTAL_FEATURE_DIM == 20
    assert len(CANONICAL_FEATURE_NAMES) == 20
    assert len(NUMERIC_FEATURE_NAMES) == 6
    assert len(CATEGORICAL_FEATURE_NAMES) == 14
    assert len(ORDERED_FAILURE_CODES) == 9
    assert len(ORDERED_PAYMENT_METHODS) == 5


def test_categorical_one_hot_encoding_properties():
    """Verify one-hot encoding has exactly one active bit per category group and no artificial ordinal relation."""
    builder = ContextBuilder()
    
    case1 = RecoveryCase(
        case_id="CASE-1",
        customer_id="CUST-1",
        amount_at_risk=1000.0,
        failure_code=PaymentFailureCode.CARD_EXPIRED,
        days_overdue=0,
    )
    cust1 = CustomerProfile(
        customer_id="CUST-1",
        customer_value=5000.0,
        subscription_age_days=100,
        previous_success_rate=1.0,
        previous_contact_count=0,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
    )

    ctx1 = builder.build_context(case1, cust1)
    enc1 = ctx1.categorical_encodings

    # Exactly 1 failure code is 1.0, all other 8 are 0.0
    fail_vals1 = [enc1[f"fail_code_{c.value}"] for c in ORDERED_FAILURE_CODES]
    assert sum(fail_vals1) == 1.0
    assert enc1["fail_code_CARD_EXPIRED"] == 1.0
    assert enc1["fail_code_INSUFFICIENT_FUNDS"] == 0.0

    # Exactly 1 payment method is 1.0, all other 4 are 0.0
    pay_vals1 = [enc1[f"pay_method_{m.value}"] for m in ORDERED_PAYMENT_METHODS]
    assert sum(pay_vals1) == 1.0
    assert enc1["pay_method_CREDIT_CARD"] == 1.0
    assert enc1["pay_method_UPI"] == 0.0

    # Check that different categories have equal geometric distance (no artificial ordinal meaning)
    case2 = RecoveryCase(
        case_id="CASE-2",
        customer_id="CUST-2",
        amount_at_risk=1000.0,
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=0,
    )
    case3 = RecoveryCase(
        case_id="CASE-3",
        customer_id="CUST-3",
        amount_at_risk=1000.0,
        failure_code=PaymentFailureCode.DO_NOT_HONOR,
        days_overdue=0,
    )

    vec1 = builder.to_numpy(builder.build_context(case1, cust1))
    vec2 = builder.to_numpy(builder.build_context(case2, cust1))
    vec3 = builder.to_numpy(builder.build_context(case3, cust1))

    dist_1_2 = np.linalg.norm(vec1 - vec2)
    dist_1_3 = np.linalg.norm(vec1 - vec3)
    dist_2_3 = np.linalg.norm(vec2 - vec3)

    # In one-hot encoding, the Euclidean distance between any two distinct failure codes is sqrt(2)
    assert np.isclose(dist_1_2, np.sqrt(2.0))
    assert np.isclose(dist_1_3, np.sqrt(2.0))
    assert np.isclose(dist_2_3, np.sqrt(2.0))


def test_deterministic_normalization_and_clipping():
    """Verify continuous features are scaled and bounded deterministically."""
    scale_cfg = FeatureScaleConfig(
        amount_at_risk_scale=10000.0,
        days_overdue_scale=90.0,
        customer_value_scale=50000.0,
        subscription_age_scale=3650.0,
        contact_count_scale=20.0,
        clip_bounds=True,
    )
    builder = ContextBuilder(schema_version="v1.1.0", scale_config=scale_cfg)

    # Case with extreme / outlier values to verify clipping
    case = RecoveryCase(
        case_id="CASE-NORM",
        customer_id="CUST-NORM",
        amount_at_risk=15000.0,  # exceeds 10000 -> clipped to 1.0
        failure_code=PaymentFailureCode.INSUFFICIENT_FUNDS,
        days_overdue=180,        # exceeds 90 -> clipped to 1.0
    )
    customer = CustomerProfile(
        customer_id="CUST-NORM",
        customer_value=25000.0,  # 25000 / 50000 = 0.5
        subscription_age_days=730,  # 730 / 3650 = 0.2
        previous_success_rate=0.8,
        previous_contact_count=5,   # 5 / 20 = 0.25
        payment_method_type=PaymentMethodType.UPI,
    )

    ctx = builder.build_context(case, customer)
    vec = builder.to_numpy(ctx)

    assert len(vec) == TOTAL_FEATURE_DIM
    assert vec[0] == 1.0    # amount_at_risk_norm (clipped)
    assert vec[1] == 1.0    # days_overdue_norm (clipped)
    assert vec[2] == 0.5    # customer_value_norm
    assert np.isclose(vec[3], 0.2)  # subscription_age_norm
    assert vec[4] == 0.8    # previous_success_rate
    assert vec[5] == 0.25   # previous_contact_count_norm


def test_zero_null_history_edge_cases():
    """Verify brand new customers with zero history produce valid finite vectors."""
    builder = ContextBuilder()
    case = RecoveryCase(
        case_id="CASE-ZERO",
        customer_id="CUST-ZERO",
        amount_at_risk=0.01,
        failure_code=PaymentFailureCode.GENERIC_DECLINE,
        days_overdue=0,
    )
    customer = CustomerProfile(
        customer_id="CUST-ZERO",
        customer_value=0.0,
        subscription_age_days=0,
        previous_success_rate=0.0,
        previous_contact_count=0,
        payment_method_type=PaymentMethodType.CREDIT_CARD,
    )

    ctx = builder.build_context(case, customer)
    vec = builder.to_numpy(ctx)

    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)


def test_reproducibility_identical_inputs():
    """Verify repeated construction with identical input produces bit-exact identical vector x."""
    builder1 = ContextBuilder(schema_version="v1.1.0")
    builder2 = ContextBuilder(schema_version="v1.1.0")

    case = RecoveryCase(
        case_id="CASE-REP",
        customer_id="CUST-REP",
        amount_at_risk=4280.0,
        failure_code=PaymentFailureCode.AUTHENTICATION_REQUIRED,
        days_overdue=5,
    )
    customer = CustomerProfile(
        customer_id="CUST-REP",
        customer_value=12000.0,
        subscription_age_days=365,
        previous_success_rate=0.65,
        previous_contact_count=3,
        payment_method_type=PaymentMethodType.DIGITAL_WALLET,
    )

    ctx1 = builder1.build_context(case, customer)
    ctx2 = builder2.build_context(case, customer)

    assert ctx1.feature_vector == ctx2.feature_vector
    assert np.array_equal(builder1.to_numpy(ctx1), builder2.to_numpy(ctx2))
