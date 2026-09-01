# System Architecture Specification -- Frozen Architecture v5

## 1. Architectural Philosophy & Authority Boundaries

The AI Revenue Recovery System enforces strict architectural separation of concerns. Every component has defined inputs, outputs, and explicit boundaries on what it is **prohibited** from doing.

```text
PaymentFailureEvent
        |
        v
RecoveryCase + CustomerProfile
        |
        v
ContextBuilder ------> ContextFeatures (x in R^20)
        |
        v
CandidateActionGenerator ----> Candidate Actions
        |
        v
PolicyEngine --------> A_allowed subset of Candidate Actions
        |
        v
TwoStepValueEngine --> Q2(s, a) = R(s, a) + gamma * sum P(s'|s,a) V1(s') - C(s, a)
        |              (Future lookahead V1 has ZERO exploration bonus)
        v
LinUCBValueModel ----> Current exploration bonus B(x, a) = alpha * sqrt(x^T A_a^-1 x)
        |
        v
DecisionEngine ------> a* = argmax [ Q2(s, a) + B(x, a) ]
        |
        v
ActionExecutor ------> Atomic Idempotency Key: (case_id, decision_id, action, attempt)
        |
        v
VerificationEngine --> Provisional State: IN_OBSERVATION (Holding Window)
        |
        v
Reconciliation ------> Final State: RESOLVED_RECOVERED / RESOLVED_UNRECOVERABLE
        |
        v
LLM Messaging -------> Candidate Message
        |
        v
MessageValidator ----> APPROVED / REJECTED (Zero LLM decision authority)
```

---

## 2. Component Authority & Responsibility Table

| Component | Responsibility | Inputs | Outputs | Prohibited Actions (Boundary) |
|---|---|---|---|---|
| **ContextBuilder** | Deterministically encodes domain entities into feature vector. | `RecoveryCase`, `CustomerProfile` | `ContextFeatures` (x in R^20) | Cannot modify case or customer records. |
| **PolicyEngine** | Evaluates compliance, PCI, fatigue, and network decline rules. | `RecoveryCase`, `CustomerProfile`, `Candidates` | `PolicyDecision` (`allowed_actions`, `prohibited_actions`) | Cannot score actions or select optimal action. |
| **LinUCBValueModel** | Predicts base value Q1(x, a) and computes exploration bonus B(x, a). | `ContextFeatures`, `ActionType` | Q1(x, a), B(x, a) | Cannot evaluate future horizon states or apply policy. |
| **TwoStepValueEngine**| Evaluates sequence value Q2(s, a) over two-step horizon. | `RecoveryCase`, `CustomerProfile`, `A_allowed` | `TwoStepScoringResult` | Cannot include exploration bonus in future V1(s'). |
| **DecisionEngine** | Performs argmax over policy-allowed actions with canonical tie-breaking. | `RecoveryCase`, `CustomerProfile`, `Candidates` | `DecisionResult`, `DecisionAuditRecord` | Cannot inject STOP into empty candidate set; cannot bypass policy. |
| **ActionExecutor** | Executes external action adapters through atomic idempotency store. | `ActionType`, `RecoveryCase`, `CustomerProfile`, `decision_id`, `attempt` | `ExecutionResult` | Cannot alter decision; cannot duplicate external side effects. |
| **VerificationEngine**| Enforces holding windows and processes settlement reconciliation. | `ExecutionResult`, `ReconciliationData` | `ObservationRecord`, final `CaseState` | Cannot mark recovery before window elapses; cannot reverse finalized states. |
| **MessageValidator** | Deterministically verifies message body against ground truth. | `CandidateMessage`, `RecoveryCase`, `CustomerProfile`, `selected_action` | `ValidationResult` (`APPROVED`/`REJECTED`) | Cannot modify `selected_action`, amount, or failure code. |
| **LLM Messaging** | Generates customer-facing explanatory messages. | `MessageTemplate`, `RecoveryCase`, `CustomerProfile`, `selected_action` | `CandidateMessage` | **ZERO decision or financial authority.** |

---

## 3. Strict Authority Invariants

1. **Policy Priority Invariant:** Policy and safety rules execute strictly *before* mathematical value evaluation.
2. **Provisional Invariant:** Execution `SUCCESS` produces provisional `IN_OBSERVATION`. It **never** marks `RESOLVED_RECOVERED` directly.
3. **Reconciliation Invariant:** Final recovery requires observation window completion and settlement confirmation without invalidating disputes.
4. **Idempotency Invariant:** Duplicate execution of composite key `(case_id, decision_id, action, attempt)` returns cached result without re-executing external adapters.
5. **Anti-Hallucination Invariant:** Message validation is deterministic and validates the actual message body against structured case ground truth.
