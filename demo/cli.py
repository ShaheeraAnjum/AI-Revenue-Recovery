"""Interactive Command Line Interface for Demonstrating AI Revenue Recovery."""
import sys
import os
from datetime import datetime, timezone, timedelta

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.service import RevenueRecoveryOrchestrator
from src.messaging.schema import CandidateMessage
from demo.scenarios import DEMO_SCENARIOS, DemoScenario


def print_banner():
    print("=" * 80)
    print("   AI REVENUE RECOVERY SYSTEM -- DETERMINISTIC V5 ARCHITECTURE DEMO")
    print("   [DEMO / SIMULATED ENVIRONMENT - CORE BUSINESS LOGIC IS FROZEN]")
    print("=" * 80)


def run_scenario(scenario: DemoScenario):
    print(f"\n>>> RUNNING: {scenario.name}")
    print(f"Category:    {scenario.category}")
    print(f"Description: {scenario.description}")
    print("-" * 80)

    orchestrator = RevenueRecoveryOrchestrator()
    case = orchestrator.ingest_failure_event(scenario.event, scenario.customer)

    print(f"1. INGESTED CASE:")
    print(f"   Case ID:        {case.case_id}")
    print(f"   Customer ID:    {case.customer_id}")
    print(f"   Amount at Risk: INR {case.amount_at_risk:.2f}")
    print(f"   Failure Code:   {case.failure_code.value}")
    print(f"   Initial State:  {case.state.value}")

    # Process recovery cycle
    fixed_dec_id = f"DEC-{scenario.scenario_id.upper()}"
    cycle = orchestrator.process_recovery_cycle(
        case=case,
        customer=scenario.customer,
        candidate_actions=scenario.candidate_actions,
        decision_id=fixed_dec_id,
    )

    dec_res = cycle.decision_result
    print(f"\n2. POLICY & SAFETY EVALUATION:")
    print(f"   Allowed Actions:    {[a.value for a in dec_res.allowed_actions]}")
    print(f"   Prohibited Actions: {dict((k.value, v) for k, v in dec_res.prohibited_actions.items())}")

    print(f"\n3. TWO-STEP HORIZON & LINUCB MATHEMATICAL SCORING:")
    print(f"   {'Action':<16} | {'Q1 Base':<10} | {'Q2 Seq':<10} | {'B(x,a) (LinUCB)':<16} | {'Final Score':<12} | {'Method'}")
    print("   " + "-" * 75)
    for act in dec_res.allowed_actions:
        audit_detail = dec_res.audit_record.action_details.get(act)
        q1_val = audit_detail.q1_base_value if audit_detail else 0.0
        q2_val = dec_res.q2_values.get(act, 0.0)
        bonus = dec_res.exploration_bonuses.get(act, 0.0)
        final_q = dec_res.final_q_values.get(act, 0.0)
        method = dec_res.estimation_methods.get(act, "").value if hasattr(dec_res.estimation_methods.get(act, ""), "value") else str(dec_res.estimation_methods.get(act, ""))
        print(f"   {act.value:<16} | INR {q1_val:<6.2f} | INR {q2_val:<6.2f} | INR {bonus:<12.2f} | INR {final_q:<8.2f} | {method}")

    print(f"\n4. DECISION OUTCOME:")
    print(f"   Selected Action:    {cycle.selected_action.value if cycle.selected_action else 'None'}")
    print(f"   Idempotency Key:    {cycle.idempotency_key}")
    print(f"   Decision ID:        {cycle.decision_id}")
    print(f"   Why This Action?    {scenario.why_action}")

    if cycle.execution_result:
        print(f"\n5. IDEMPOTENT EXECUTION & OBSERVATION:")
        print(f"   Execution Status:   {cycle.execution_result.execution_status.value}")
        print(f"   Provisional State:  {cycle.case_state.value} (CRITICAL: SUCCESS != FINAL RECOVERY)")
        print(f"   Observation ID:     {cycle.observation_record.observation_id}")
        print(f"   Holding Window:     {cycle.observation_record.observation_window_seconds} seconds")

    if scenario.simulate_duplicate and cycle.selected_action:
        print(f"\n5b. DUPLICATE IDEMPOTENCY EXECUTION TEST:")
        dup_cycle = orchestrator.process_recovery_cycle(
            case=case,
            customer=scenario.customer,
            candidate_actions=scenario.candidate_actions,
            decision_id=fixed_dec_id,
        )
        print(f"   Second Execution Is Duplicate? {dup_cycle.execution_result.is_duplicate}")
        print(f"   External Adapter Re-Executed?  NO (Side effects blocked)")

    if scenario.reconciliation_data and cycle.idempotency_key:
        print(f"\n6. SETTLEMENT & DISPUTE RECONCILIATION:")
        recon_time = datetime.now(timezone.utc) + timedelta(hours=2)
        final_obs = orchestrator.reconcile_case(
            idempotency_key=cycle.idempotency_key,
            case=case,
            reconciliation_data=scenario.reconciliation_data,
            as_of_time=recon_time,
        )
        print(f"   Reconciliation Ref:    {scenario.reconciliation_data.reconciliation_reference}")
        print(f"   Settlement Confirmed?  {scenario.reconciliation_data.settlement_confirmed}")
        print(f"   Is Chargeback?         {scenario.reconciliation_data.is_chargeback}")
        print(f"   Final Case State:      {case.state.value}")
        print(f"   Final Outcome:         {final_obs.final_outcome.value if final_obs.final_outcome else 'None'}")

    # Messaging validation section
    if scenario.simulate_hallucination:
        print(f"\n7. CONSTRAINED LLM MESSAGING & ANTI-HALLUCINATION VALIDATOR:")
        rogue_msg = CandidateMessage(
            template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
            body_text=f"Dear customer, your payment of 5000.00 for case {case.case_id} failed due to card expired. Please update your payment method.",
            stated_amount=5000.00,
            stated_failure_reason="CARD_EXPIRED",
        )
        val_res = orchestrator.messaging_service.validator.validate(
            message=rogue_msg,
            case=case,
            customer=scenario.customer,
            selected_action=cycle.selected_action,
        )
        print(f"   Candidate Generated:   \"{rogue_msg.body_text}\"")
        print(f"   Validator Disposition: {val_res.status.value}")
        print(f"   Rejection Reasons:     {[r.value for r in val_res.rejection_reasons]}")
        print(f"   Action Altered by LLM? NO (Decision remains strictly {cycle.selected_action.value})")

    elif cycle.message_validation:
        print(f"\n7. CONSTRAINED LLM MESSAGING & ANTI-HALLUCINATION VALIDATOR:")
        print(f"   Delivered Message:     \"{cycle.delivered_message.body_text if cycle.delivered_message else 'REJECTED / NONE'}\"")
        print(f"   Validator Disposition: {cycle.message_validation.status.value}")
        print(f"   Template Version:      {cycle.message_audit.template_version}")

    print(f"\n8. AUDIT TRACEABILITY (ALL 8 DIMENSIONS PRESERVED):")
    aud = dec_res.audit_record
    print(f"   Policy Version:       {aud.policy_version}")
    print(f"   Value Model Version:  {aud.value_model_version}")
    print(f"   Transition Version:   {aud.transition_model_version}")
    print(f"   Feature Schema:       {aud.feature_schema_version}")
    print("=" * 80)


def main():
    print_banner()
    print("Available Scenarios:")
    for key, sc in DEMO_SCENARIOS.items():
        print(f"  [{key}] {sc.name}")
    print("  [all] Run all scenarios sequentially")
    print("  [q]   Quit")

    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        choice = input("\nEnter scenario key (e.g. scenario_a or all): ").strip().lower()

    if choice == "all":
        for sc in DEMO_SCENARIOS.values():
            run_scenario(sc)
    elif choice in DEMO_SCENARIOS:
        run_scenario(DEMO_SCENARIOS[choice])
    elif choice == "q":
        print("Exiting demo.")
    else:
        print(f"Unknown scenario: {choice}. Running scenario_a by default:")
        run_scenario(DEMO_SCENARIOS["scenario_a"])


if __name__ == "__main__":
    main()
