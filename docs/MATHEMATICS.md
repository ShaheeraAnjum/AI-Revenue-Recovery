# Mathematical Foundations -- Frozen Architecture v5

This document details the exact mathematical formulations implemented in the codebase.

---

## 1. Contextual LinUCB Value Model

For every supported action a in {RETRY, PAYMENT_UPDATE, REMINDER, WAIT}:

### A. Point Value Prediction (Base Value Q1)
$$Q_1(x, a) = x^T \theta_a$$
where:
- $x \in \mathbb{R}^{20}$ is the normalized context feature vector.
- $\theta_a = A_a^{-1} b_a \in \mathbb{R}^{20}$ is the parameter vector for action a.
- $A_a = I_{20} + \sum_{i=1}^N x_i x_i^T \in \mathbb{R}^{20 \times 20}$ is the design matrix initialized with identity regularization.
- $b_a = \sum_{i=1}^N r_i x_i \in \mathbb{R}^{20}$ is the accumulated response vector.

### B. LinUCB Exploration Bonus
$$B(x, a) = \alpha \sqrt{x^T A_a^{-1} x}$$
- $\alpha \ge 0.0$ is the exploration scaling hyperparameter.
- The **square root** is strictly mandatory and represents the standard deviation of the ridge regression estimate.
- For restricted action `ESCALATE`: $B(x, \text{ESCALATE}) = 0.0$ (estimation method: `heuristic`).
- For baseline action `STOP`: $B(x, \text{STOP}) = 0.0$ and $Q_1(x, \text{STOP}) = 0.0$.

---

## 2. Two-Step Horizon Value Formulation (Q2)

The sequence value $Q_2(s, a)$ models the immediate reward, dynamic intervention cost, and expected future value over a two-step lookahead horizon:

$$Q_2(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a, x) V_1(s') - C(s, a)$$

where:
1. **Immediate Reward $R(s, a)$:**
   $$R(s, a) = P(s' = \text{RECOVERED} \mid s, a, x) \times \text{amount\_at\_risk}$$
2. **Discount Factor $\gamma$:** $\gamma \in [0, 1]$ (default: $0.95$).
3. **Transition Probability $P(s' \mid s, a, x)$:** Action-conditional transition probability distribution over state space $S = \{\text{IN\_OBSERVATION}, \text{RECOVERED}, \text{UNRECOVERABLE}\}$.
4. **Future Lookahead Value $V_1(s')$ (CRITICAL INVARIANT):**
   $$V_1(s') = \begin{cases} 
   0.0 & \text{if } s' \in \{\text{RECOVERED}, \text{UNRECOVERABLE}\} \\
   \max_{a' \in A_{\text{allowed}}(s')} Q_1(s', a') & \text{if } s' = \text{IN\_OBSERVATION}
   \end{cases}$$
   > **ZERO EXPLORATION BONUS IN LOOKAHEAD:** $V_1(s')$ strictly evaluates base value $Q_1(s', a')$ and **never** includes the exploration bonus $B(x', a')$.
5. **Action Cost $C(s, a)$:**
   - For `RETRY`, `PAYMENT_UPDATE`, `REMINDER`, `ESCALATE`, `STOP`: configured static operational costs.
   - For `WAIT`: Dynamic holding cost formula.

---

## 3. Dynamic Wait Cost Formulation

To prevent indefinite delay and penalize aging arrears, the cost of `WAIT` is dynamically calculated as:

$$C_{\text{wait}} = r_{\text{hold}} \times \text{days\_waiting} + r_{\text{delay}} \times \text{days\_overdue}$$

where:
- $r_{\text{hold}} \ge 0.0$: daily holding cost rate for consecutive days waiting in recovery.
- $r_{\text{delay}} \ge 0.0$: delay depreciation penalty rate proportional to invoice overdue aging.

---

## 4. Current Action Decision Formula (Argmax)

The final optimal action $a^*$ is chosen from the policy-approved candidate set $A_{\text{allowed}}$ via single argmax:

$$a^* = \arg\max_{a \in A_{\text{allowed}}} \left[ Q_2(s, a) + B(x, a) \right]$$

### Deterministic Tie-Breaking Priority
In the event of identical final scores, ties are resolved deterministically using canonical priority:
$$\text{RETRY} > \text{PAYMENT\_UPDATE} > \text{REMINDER} > \text{WAIT} > \text{ESCALATE} > \text{STOP}$$
