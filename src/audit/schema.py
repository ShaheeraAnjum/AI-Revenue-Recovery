"""Audit and reproducibility schema capturing all 8 required version dimensions."""
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from src.domain.actions import ActionType


class EstimationMethod(str, Enum):
    """Explicit estimation provenance tagging."""
    CONTEXTUAL_BANDIT = "contextual_bandit"
    HEURISTIC = "heuristic"


class VersionConfig(BaseModel):
    """The 8 mandatory version dimensions required by frozen v5 architecture."""
    policy_version: str = "policy_v5.0.0"
    value_model_version: str = "linucb_v5.0.0"
    transition_model_version: str = "trans_v5.0.0"
    propensity_model_version: str = "prop_v5.0.0"
    fairness_policy_version: str = "fair_v5.0.0"
    message_policy_version: str = "msg_v5.0.0"
    feature_schema_version: str = "feat_v1.0.0"
    exploration_config_version: str = "exp_v5.0.0"


class ActionScoringDetail(BaseModel):
    """Detailed breakdown for each candidate/evaluated action."""
    action: ActionType
    is_allowed: bool
    rejection_reason: Optional[str] = None
    estimation_method: Optional[EstimationMethod] = None
    q1_base_value: Optional[float] = None
    q2_sequence_value: Optional[float] = None
    exploration_bonus: Optional[float] = Field(
        default=None,
        description="LinUCB exploration bonus B(x, a) = alpha * sqrt(x^T A_a^-1 x)"
    )
    final_q_value: Optional[float] = None
    confidence: Optional[float] = None


class DecisionAuditRecord(BaseModel):
    """Complete audit log of a single recovery decision for 100% offline reproducibility."""
    decision_id: str
    case_id: str
    customer_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Feature context
    context_features: Dict[str, float]
    feature_vector: List[float]
    
    # Candidate action generation & policy filtering
    candidate_actions: List[ActionType]
    allowed_actions: List[ActionType]
    prohibited_actions: Dict[ActionType, str]
    
    # Mathematical values & scores
    action_details: Dict[ActionType, ActionScoringDetail]
    q1_values: Dict[ActionType, float]
    q2_values: Dict[ActionType, float]
    exploration_bonuses: Dict[ActionType, float] = Field(
        ...,
        description="Exploration bonus B(x, a) = alpha * sqrt(x^T A_a^-1 x) for each evaluated action"
    )
    final_q_values: Dict[ActionType, float]
    estimation_methods: Dict[ActionType, EstimationMethod]
    confidences: Dict[ActionType, float]
    
    # Selected action
    selected_action: ActionType
    random_seed: Optional[int] = None
    
    # The 8 mandatory version dimensions
    policy_version: str
    value_model_version: str
    transition_model_version: str
    propensity_model_version: str
    fairness_policy_version: str
    message_policy_version: str
    feature_schema_version: str
    exploration_config_version: str
    
    # Extra metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
