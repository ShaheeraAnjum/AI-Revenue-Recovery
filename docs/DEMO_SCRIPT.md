# Live Demonstration Script

Use this step-by-step walkthrough script during academic, faculty, or technical project evaluations.

---

## Step 1: Start the Interactive Web UI Dashboard
1. Open a terminal in the project directory.
2. Execute:
   ```bash
   python demo/server.py
   ```
3. Open **[http://localhost:8080](http://localhost:8080)** in the web browser.
4. **Key Talking Point:** Note that the demo server runs with zero external pip dependencies using standard library modules and operates completely offline with simulated data.

---

## Step 2: Demonstrate Scenario A -- Happy Path Recovery
1. In the Web UI, click on **Scenario A: Standard Insufficient Funds**.
2. **Observe & Explain:**
   - **Stage 1 (Policy):** Policy permits RETRY, PAYMENT_UPDATE, REMINDER, WAIT, STOP. ESCALATE is pruned (insufficient overdue aging).
   - **Stage 2 (Scoring):** Point out the mathematical table showing Q1, Q2, and the LinUCB exploration bonus B(x,a) = alpha * sqrt(x^T A_a^-1 x).
   - **Stage 3 (Execution):** Note the provisional state is IN_OBSERVATION. Explain that execution success does **not** directly equal recovery.
   - **Stage 4 (Reconciliation):** Show the confirmed settlement ledger transaction transitioning the case to RESOLVED_RECOVERED.

---

## Step 3: Demonstrate Scenario C -- Hard Policy Decline (Fraud Suspected)
1. Click on **Scenario C: Fraud Suspected**.
2. **Observe & Explain:**
   - **Policy Engine Pruning:** Because the failure code is FRAUD_SUSPECTED, hard policy rules prohibit RETRY and ESCALATE under active case conditions.
   - **Decision Engine:** The Decision Engine evaluates the remaining permitted actions and selects PAYMENT_UPDATE based on the highest valid score. Unsafe messages failing validation requirements are rejected.

---

## Step 4: Demonstrate Scenario G -- Anti-Hallucination Message Validator
1. Click on **Scenario G: Rogue LLM Hallucination Rejection**.
2. **Observe & Explain:**
   - Show how a simulated rogue LLM candidate generated a message with hallucinated values (claimed INR 5,000 when actual was INR 1,200, and claimed CARD_EXPIRED when actual was INSUFFICIENT_FUNDS).
   - Show MessageValidator catching the error and setting disposition to REJECTED with codes AMOUNT_MISMATCH and FAILURE_CODE_MISMATCH.
   - Highlight the **critical invariant**: The Decision Engine selected action (PAYMENT_UPDATE) remains **100% unchanged**. The LLM has zero decision authority.

---

## Step 5: Demonstrate Scenario H -- Atomic Idempotency Guard
1. Click on **Scenario H: Concurrent Duplicate Request**.
2. **Observe & Explain:**
   - The exact composite 4-tuple key (case_id, decision_id, action, attempt) is evaluated.
   - The second execution is flagged as is_duplicate = True, and the cached response is returned without executing duplicate external transactions.

---

## Step 6: Verify Test Suite in Terminal
1. Open a terminal and run:
   ```bash
   python -m pytest tests/ -v
   ```
2. Point out that all **124 tests pass** across all 10 architectural phases with zero errors.
