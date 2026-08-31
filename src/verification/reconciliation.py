"""Financial settlement, refund, and chargeback reconciliation records."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

DEFAULT_RECONCILIATION_VERSION: str = "recon_v5.0.0"


class ReconciliationConflictError(Exception):
    """Raised when conflicting reconciliation data is submitted for an already-finalized observation."""
    pass


class ReconciliationData(BaseModel):
    """External financial settlement and dispute ledger entry for reconciliation."""
    reconciliation_reference: str
    settlement_confirmed: bool = False
    is_refunded: bool = False
    is_chargeback: bool = False
    gross_amount_settled: float = 0.0
    net_amount_recovered: float = 0.0
    settled_at: Optional[datetime] = None
    reconciliation_version: str = DEFAULT_RECONCILIATION_VERSION
    details: Dict[str, Any] = Field(default_factory=dict)

    def compute_fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint of financially relevant reconciliation payload."""
        payload = {
            "reconciliation_reference": self.reconciliation_reference,
            "settlement_confirmed": self.settlement_confirmed,
            "is_refunded": self.is_refunded,
            "is_chargeback": self.is_chargeback,
            "gross_amount_settled": float(self.gross_amount_settled),
            "net_amount_recovered": float(self.net_amount_recovered),
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
            "reconciliation_version": self.reconciliation_version,
            "details": self.details,
        }
        canonical_json = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
