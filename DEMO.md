# AI Revenue Recovery System -- Demonstration Guide

## Overview
This document provides instructions for running and evaluating the demonstration layer for the **AI Revenue Recovery System (Frozen Architecture v5)**.

> **CRITICAL DISCLAIMER:**  
> All demo transactions, customers, invoice IDs, and gateway responses are **SIMULATED / DEMO DATA**. No live financial credentials or payment networks are connected. The underlying algorithmic decision engine, policy rules, and mathematical value models are **100% deterministic, frozen, and verified**.

---

## 1. Quick Start

### Option A: Interactive Web UI Dashboard
To launch the zero-dependency local browser dashboard:
```bash
python demo/server.py
```
Then open [http://localhost:8080](http://localhost:8080) in your web browser.

### Option B: Interactive Terminal CLI
To run the interactive CLI demonstration:
```bash
python demo/cli.py
```
Or execute a specific scenario directly:
```bash
python demo/cli.py scenario_a
python demo/cli.py all
```

---

## 2. Demonstration Scenarios

| Scenario Key | Scenario Name | Architectural Boundary Demonstrated | Expected Action & Outcome |
|---|---|---|---|
| `scenario_a` | **Standard Insufficient Funds** | Soft decline evaluated via two-step horizon scoring and LinUCB exploration. | Selected: `PAYMENT_UPDATE`. Execution `SUCCESS` -> `IN_OBSERVATION`. Settlement confirmed -> `RESOLVED_RECOVERED`. |
| `scenario_b` | **Card Expired** | Policy blocks retries. Downstream constrained messaging with verified body facts. | Selected: `PAYMENT_UPDATE`. Validated email generated and delivered. |
| `scenario_c` | **Suspected Fraud / Hard Decline** | Hard policy rules prohibit RETRY and ESCALATE under active case conditions. | Selected: `PAYMENT_UPDATE` (highest valid permitted score). Unsafe messages rejected. |
| `scenario_d` | **Contact Fatigue Limit** | Customer contact frequency limit reached. Dynamic wait cost calculated (C_wait). | Selected: `WAIT`. Holding cost dynamically evaluated. |
| `scenario_e` | **Negative Sequence Value (STOP)** | Interventions produce negative expected return; baseline Q(STOP)=0.0 wins. | Selected: `STOP`. Transition to `RESOLVED_UNRECOVERABLE`. |
| `scenario_f` | **Post-Settlement Chargeback** | Initial execution success followed by downstream bank chargeback dispute. | Reconciled to `RESOLVED_UNRECOVERABLE`. |
| `scenario_g` | **Rogue LLM Anti-Hallucination** | Rogue LLM candidate claims incorrect amount and failure reason. | `MessageValidator` detects `FAILURE_CODE_MISMATCH` & `AMOUNT_MISMATCH`. Decision action remains unchanged. |
| `scenario_h` | **Concurrent Duplicate Execution** | Duplicate request submitted with same composite idempotency key. | Safely deduplicated; external side effect executed exactly once. |

---

## 3. Architectural Boundaries Verified in Demo

1. **Policy Priority:** Safety and regulatory rules prune prohibited actions *before* mathematical scoring.
2. **Exploration Boundary:** LinUCB exploration bonus $B(x,a) = \alpha \sqrt{x^T A_a^{-1} x}$ applies *only* to current action selection. Future lookahead $V_1(s') = \max_{a'} Q_1(s', a')$ has strictly zero exploration bonus.
3. **Provisional Invariant:** Execution `SUCCESS` directly produces `IN_OBSERVATION` and never marks `RESOLVED_RECOVERED` without ledger settlement reconciliation.
4. **Zero LLM Authority:** The LLM cannot select actions, alter amounts, or determine recovery.

---

## 4. Verifying Frozen Test Suite

The demo layer does not modify any core files. You can verify that all 124 frozen tests continue to pass:
```bash
python -m pytest tests/ -v
```
Expected output: **124 passed in ~1.4s**.
