"""Action definitions for AI Revenue Recovery."""
from enum import Enum


class ActionType(str, Enum):
    """Candidate actions defined in the v5 frozen architecture."""
    RETRY = "RETRY"
    PAYMENT_UPDATE = "PAYMENT_UPDATE"
    REMINDER = "REMINDER"
    WAIT = "WAIT"
    ESCALATE = "ESCALATE"
    STOP = "STOP"


class ActionCategory(str, Enum):
    """Categorization of actions based on causal learning support."""
    SUPPORTED = "SUPPORTED"
    RESTRICTED = "RESTRICTED"
    TERMINAL = "TERMINAL"


SUPPORTED_ACTIONS = {
    ActionType.RETRY,
    ActionType.PAYMENT_UPDATE,
    ActionType.REMINDER,
    ActionType.WAIT,
}

RESTRICTED_ACTIONS = {
    ActionType.ESCALATE,
}

TERMINAL_ACTIONS = {
    ActionType.STOP,
}


def is_supported_action(action: ActionType) -> bool:
    """Return True if action belongs to the supported LinUCB learning region."""
    return action in SUPPORTED_ACTIONS


def is_restricted_action(action: ActionType) -> bool:
    """Return True if action is restricted and requires human-calibrated heuristic scoring."""
    return action in RESTRICTED_ACTIONS
