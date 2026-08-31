"""Deterministic generator for candidate recovery actions."""
from typing import List
from src.domain.actions import ActionType
from src.domain.case import RecoveryCase, CustomerProfile

# All 6 canonical actions defined in frozen v5 architecture
ALL_CANDIDATE_ACTIONS: List[ActionType] = [
    ActionType.RETRY,
    ActionType.PAYMENT_UPDATE,
    ActionType.REMINDER,
    ActionType.WAIT,
    ActionType.ESCALATE,
    ActionType.STOP,
]


class CandidateActionGenerator:
    """Generates the initial candidate action set before policy/safety filtering."""

    def __init__(self, actions: List[ActionType] | None = None):
        self.actions = actions or list(ALL_CANDIDATE_ACTIONS)

    def generate_candidates(
        self,
        case: RecoveryCase,
        customer: CustomerProfile,
    ) -> List[ActionType]:
        """Generate the complete list of candidate actions to be submitted to the Policy Engine."""
        return list(self.actions)
