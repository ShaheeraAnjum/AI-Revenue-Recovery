"""Financial settlement, refund, and chargeback reconciliation records."""
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

DEFAULT_RECONCILIATION_VERSION: str = "recon_v5.0.0"


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
