"""Centralized Policy & Safety Engine evaluating candidate actions."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile
from src.actions.generator import CandidateActionGenerator, ALL_CANDIDATE_ACTIONS
from src.policy.config import PolicyConfig, DEFAULT_POLICY_VERSION
from src.policy.rules import check_action_compliance


class PolicyDecision(BaseModel):
    """Output of the Policy Engine containing allowed actions and explicit rejection reasons."""
    case_id: str
    customer_id: str
    policy_version: str
    candidate_actions: List[ActionType]
    allowed_actions: List[ActionType]
    prohibited_actions: Dict[ActionType, str] = Field(
        ...,
        description="Map of rejected actions to their deterministic rejection reasons"
    )

    def is_action_allowed(self, action: ActionType) -> bool:
        """Check if an action is in the allowed set."""
        return action in self.allowed_actions


class PolicyEngine:
    """Safety and compliance engine executing before value modeling.
    Filters candidate actions into compliance-approved allowed_actions.
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
        generator: CandidateActionGenerator | None = None,
    ):
        self.config = config or PolicyConfig()
        self.generator = generator or CandidateActionGenerator()

    def evaluate(
        self,
        case: RecoveryCase,
        customer: CustomerProfile,
        candidate_actions: Optional[List[ActionType]] = None,
    ) -> PolicyDecision:
        """Deterministically evaluate candidate actions against compliance rules.
        Does NOT score, rank, compute Q-values, or call LLMs.
        """
        candidates = candidate_actions or self.generator.generate_candidates(case, customer)
        
        allowed: List[ActionType] = []
        prohibited: Dict[ActionType, str] = {}

        for action in candidates:
            is_allowed, reason = check_action_compliance(
                action=action,
                case=case,
                customer=customer,
                config=self.config,
            )
            if is_allowed:
                allowed.append(action)
            else:
                prohibited[action] = reason or "prohibited_by_policy"

        return PolicyDecision(
            case_id=case.case_id,
            customer_id=customer.customer_id,
            policy_version=self.config.policy_version,
            candidate_actions=list(candidates),
            allowed_actions=allowed,
            prohibited_actions=prohibited,
        )
