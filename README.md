# AI Revenue Recovery System

**Deterministic Two-Step Value Horizon Engine, Contextual LinUCB, Policy Guardrails, and Constrained LLM Messaging**

---

## 1. Project Overview & Problem Statement

In subscription-based and digital recurring billing environments, up to 10-20% of payments fail due to transient technical errors, temporary lack of funds, expired credentials, or bank fraud blocks (involuntary churn). Traditional recovery engines rely on static, aggressive retry rules or uncontrolled LLMs that hallucinate facts and violate compliance rules.

The **AI Revenue Recovery System (Frozen Architecture v5)** is a mathematically rigorous, compliance-hardened recovery engine that optimizes net recovery value while preserving customer trust and enforcing strict regulatory guardrails.

---

## 2. Key Capabilities

- **20-Dimensional Contextual Feature Extraction:** Deterministically extracts behavioral, financial, fatigue, and subscription signals into a normalized context vector x in R^20.
- **Pre-Scoring Policy & Safety Layer:** Hard rules (PCI tokenization, network hard decline pruning, contact frequency limits, VIP protection) filter actions *before* value evaluation.
- **Contextual LinUCB Exploration:** Exact mathematical upper confidence bound B(x,a) = alpha * sqrt(x^T A_a^-1 x) for supported actions (RETRY, PAYMENT_UPDATE, REMINDER, WAIT).
- **Two-Step Horizon Value Engine:** Evaluates Q2(s, a) = R(s, a) + gamma * sum P(s'|s, a) V1(s') - C(s, a), where future lookahead V1(s') = max Q1(s', a') carries **strictly zero exploration bonus**.
- **Dynamic Cadence & Holding Cost:** Dynamic wait penalty C_wait = r_hold * days_waiting + r_delay * days_overdue.
- **Atomic Idempotent Execution:** Guarantees that the composite key (case_id, decision_id, action, attempt) executes external side effects exactly once under concurrent workloads.
- **Provisional Observation & Settlement Reconciliation:** Execution success produces provisional IN_OBSERVATION state; final recovery (RESOLVED_RECOVERED) requires observation window completion and settlement ledger verification.
- **Constrained LLM Messaging & Anti-Hallucination Validator:** LLM operates strictly as an explanation/messaging layer with **zero decision authority**. The deterministic MessageValidator validates canonical failure phrases and monetary amounts against ground truth.
- **Full 8-Version Audit Logging:** Every decision captures all 8 subsystem versions and random_seed for complete auditability.

---

## 3. End-to-End System Pipeline

```text
Payment Failure Event
        |
   Recovery Case
        |
  Context Builder (20-Dim Vector)
        |
Candidate Actions Generator
        |
Policy / Safety Engine (Pre-Decision Pruning)
        |
Two-Step Horizon Engine (Q2 Sequence Value)
        |
Contextual LinUCB (Current Exploration Bonus B(x,a))
        |
Argmax Action Selection [Q2(s,a) + B(x,a)]
        |
Idempotent Action Executor (Atomic 4-Tuple Key)
        |
Provisional Observation (Holding Window)
        |
Settlement & Dispute Reconciliation
        |
Final Outcome (RESOLVED_RECOVERED / RESOLVED_UNRECOVERABLE)
        |
Constrained LLM Messaging & Anti-Hallucination MessageValidator
```

---

## 4. Quick Start & Demonstration

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- pydantic (v2.x), numpy, pytest

### A. Run Interactive Web UI Dashboard
```bash
python demo/server.py
```
Open **[http://localhost:8080](http://localhost:8080)** in any web browser.

### B. Run Interactive Terminal CLI
```bash
python demo/cli.py
```
Run all 8 scenarios sequentially:
```bash
python demo/cli.py all
```

### C. Demonstration Scenarios Available
1. **Scenario A:** Standard Insufficient Funds (Auto-Retry & Settlement)
2. **Scenario B:** Card Expired (Constrained Messaging & Update)
3. **Scenario C:** Fraud Suspected (Hard Policy Decline & Escalation)
4. **Scenario D:** Contact Fatigue Capping (Dynamic Wait Cost)
5. **Scenario E:** Negative Sequence Value (Terminal STOP Action)
6. **Scenario F:** Post-Settlement Chargeback (Reconciliation Invalidation)
7. **Scenario G:** Rogue LLM Anti-Hallucination (Validator Guardrail)
8. **Scenario H:** Concurrent Duplicate Request (Atomic Idempotency Guard)

---

## 5. Verification & Test Suite

Run the complete test suite:
```bash
python -m pytest tests/ -v
```
**Test Result:** **124 passed, 0 failed, 0 skipped, 0 xfailed** across all 10 phases.

---

## 6. Disclaimers & Safety Boundaries

> **DEMO / SIMULATED ENVIRONMENT DISCLAIMER:**  
> All demo customer profiles, credit card references, transaction amounts, and gateway outcomes are **SIMULATED**. No live financial credentials or payment networks are connected.

- **Zero LLM Decision Authority:** The LLM cannot select actions, alter amounts, change failure codes, or decide recovery.
- **Offline Operation:** The entire test suite and demonstration run 100% offline without external network dependencies.
