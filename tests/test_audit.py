"""Unit tests for DecisionAuditRecord schema and 8-version tracking."""
from datetime import datetime, timezone
import pytest
from src.domain.actions import ActionType
from src.audit.schema import (
    EstimationMethod,
    VersionConfig,
    DecisionAuditRecord,
    ActionScoringDetail,
)


def test_decision_audit_record_all_8_version_fields():
    """Verify that all 8 version dimensions are required and serialized in DecisionAuditRecord."""
    record = DecisionAuditRecord(
        decision_id="DEC-999",
        case_id="CASE-999",
        customer_id="CUST-999",
        timestamp=datetime.now(timezone.utc),
        context_features={"amount_at_risk": 4280.0, "days_overdue": 2.0},
        feature_vector=[4280.0, 0.0, 2.0, 5000.0, 120.0, 0.9, 1.0, 0.0],
        candidate_actions=[ActionType.RETRY, ActionType.WAIT, ActionType.ESCALATE, ActionType.STOP],
        allowed_actions=[ActionType.RETRY, ActionType.WAIT, ActionType.STOP],
        prohibited_actions={ActionType.ESCALATE: "Max escalation policy reached"},
        action_details={
            ActionType.RETRY: ActionScoringDetail(
                action=ActionType.RETRY,
                is_allowed=True,
                estimation_method=EstimationMethod.CONTEXTUAL_BANDIT,
                q1_base_value=4200.0,
                q2_sequence_value=4150.0,
                exploration_bonus=30.0,  # alpha * sqrt(x^T A_a^-1 x)
                final_q_value=4180.0,
                confidence=0.91,
            ),
            ActionType.STOP: ActionScoringDetail(
                action=ActionType.STOP,
                is_allowed=True,
                estimation_method=EstimationMethod.CONTEXTUAL_BANDIT,
                q1_base_value=0.0,
                q2_sequence_value=0.0,
                exploration_bonus=0.0,
                final_q_value=0.0,
                confidence=1.0,
            ),
        },
        q1_values={ActionType.RETRY: 4200.0, ActionType.STOP: 0.0},
        q2_values={ActionType.RETRY: 4150.0, ActionType.STOP: 0.0},
        exploration_bonuses={ActionType.RETRY: 30.0, ActionType.STOP: 0.0},
        final_q_values={ActionType.RETRY: 4180.0, ActionType.STOP: 0.0},
        estimation_methods={
            ActionType.RETRY: EstimationMethod.CONTEXTUAL_BANDIT,
            ActionType.STOP: EstimationMethod.CONTEXTUAL_BANDIT,
        },
        confidences={ActionType.RETRY: 0.91, ActionType.STOP: 1.0},
        selected_action=ActionType.RETRY,
        random_seed=42,
        
        # 8 mandatory version dimensions
        policy_version="policy_v5.0.0",
        value_model_version="linucb_v5.0.0",
        transition_model_version="trans_v5.0.0",
        propensity_model_version="prop_v5.0.0",
        fairness_policy_version="fair_v5.0.0",
        message_policy_version="msg_v5.0.0",
        feature_schema_version="feat_v1.0.0",
        exploration_config_version="exp_v5.0.0",
    )

    data = record.model_dump()
    
    assert data["policy_version"] == "policy_v5.0.0"
    assert data["value_model_version"] == "linucb_v5.0.0"
    assert data["transition_model_version"] == "trans_v5.0.0"
    assert data["propensity_model_version"] == "prop_v5.0.0"
    assert data["fairness_policy_version"] == "fair_v5.0.0"
    assert data["message_policy_version"] == "msg_v5.0.0"
    assert data["feature_schema_version"] == "feat_v1.0.0"
    assert data["exploration_config_version"] == "exp_v5.0.0"
    
    assert data["selected_action"] == "RETRY"
    assert data["random_seed"] == 42
    assert data["estimation_methods"]["RETRY"] == "contextual_bandit"
