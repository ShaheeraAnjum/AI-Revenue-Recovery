"""Zero-dependency interactive Web UI server for demonstrating the AI Revenue Recovery System."""
import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.actions import ActionType
from src.domain.case import PaymentFailureCode, PaymentMethodType
from src.domain.events import PaymentFailureEvent
from src.orchestration.service import RevenueRecoveryOrchestrator
from src.messaging.schema import CandidateMessage
from demo.scenarios import DEMO_SCENARIOS, DemoScenario

PORT = 8080


class DemoRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving demo API and static dashboard."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self.serve_static_html()
        elif path == "/api/scenarios":
            self.serve_scenarios()
        elif path.startswith("/api/run/"):
            scenario_id = path.split("/")[-1]
            self.serve_run_scenario(scenario_id)
        else:
            self.send_error(404, "File not found")

    def serve_static_html(self):
        html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        else:
            self.send_error(404, "Dashboard HTML not found")

    def serve_scenarios(self):
        sc_list = []
        for sid, sc in DEMO_SCENARIOS.items():
            sc_list.append({
                "scenario_id": sid,
                "name": sc.name,
                "category": sc.category,
                "description": sc.description,
                "amount": sc.event.amount,
                "failure_code": sc.event.failure_code.value,
            })
        self.send_json({"scenarios": sc_list})

    def serve_run_scenario(self, scenario_id: str):
        if scenario_id not in DEMO_SCENARIOS:
            self.send_json({"error": f"Scenario {scenario_id} not found"}, status=404)
            return

        sc = DEMO_SCENARIOS[scenario_id]
        orchestrator = RevenueRecoveryOrchestrator()
        case = orchestrator.ingest_failure_event(sc.event, sc.customer)

        fixed_dec_id = f"DEC-{sc.scenario_id.upper()}"
        cycle = orchestrator.process_recovery_cycle(
            case=case,
            customer=sc.customer,
            candidate_actions=sc.candidate_actions,
            decision_id=fixed_dec_id,
        )

        dec_res = cycle.decision_result
        scoring_table = []
        for act in dec_res.allowed_actions:
            detail = dec_res.audit_record.action_details.get(act)
            scoring_table.append({
                "action": act.value,
                "q1_base_value": round(detail.q1_base_value, 2) if detail else 0.0,
                "q2_sequence_value": round(dec_res.q2_values.get(act, 0.0), 2),
                "exploration_bonus": round(dec_res.exploration_bonuses.get(act, 0.0), 2),
                "final_score": round(dec_res.final_q_values.get(act, 0.0), 2),
                "estimation_method": str(dec_res.estimation_methods.get(act, "").value if hasattr(dec_res.estimation_methods.get(act, ""), "value") else dec_res.estimation_methods.get(act, "")),
            })

        recon_result = None
        if sc.reconciliation_data and cycle.idempotency_key:
            recon_time = datetime.now(timezone.utc) + timedelta(hours=2)
            final_obs = orchestrator.reconcile_case(
                idempotency_key=cycle.idempotency_key,
                case=case,
                reconciliation_data=sc.reconciliation_data,
                as_of_time=recon_time,
            )
            recon_result = {
                "reference": sc.reconciliation_data.reconciliation_reference,
                "settlement_confirmed": sc.reconciliation_data.settlement_confirmed,
                "is_chargeback": sc.reconciliation_data.is_chargeback,
                "final_outcome": final_obs.final_outcome.value if final_obs.final_outcome else "None",
                "final_case_state": case.state.value,
            }

        hallucination_demo = None
        if sc.simulate_hallucination and cycle.selected_action:
            rogue_msg = CandidateMessage(
                template_id="TMPL_PAYMENT_UPDATE_EMAIL_V1",
                body_text=f"Dear customer, your payment of 5000.00 for case {case.case_id} failed due to card expired. Please update.",
                stated_amount=5000.00,
                stated_failure_reason="CARD_EXPIRED",
            )
            val_res = orchestrator.messaging_service.validator.validate(
                message=rogue_msg,
                case=case,
                customer=sc.customer,
                selected_action=cycle.selected_action,
            )
            hallucination_demo = {
                "candidate_text": rogue_msg.body_text,
                "status": val_res.status.value,
                "rejection_reasons": [r.value for r in val_res.rejection_reasons],
                "action_preserved": cycle.selected_action.value,
            }

        response_payload = {
            "scenario": {
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "category": sc.category,
                "description": sc.description,
            },
            "case": {
                "case_id": case.case_id,
                "customer_id": case.customer_id,
                "amount_at_risk": case.amount_at_risk,
                "failure_code": case.failure_code.value,
                "initial_state": "ACTIVE",
                "provisional_state": cycle.case_state.value,
            },
            "policy": {
                "allowed_actions": [a.value for a in dec_res.allowed_actions],
                "prohibited_actions": {k.value: v for k, v in dec_res.prohibited_actions.items()},
                "policy_version": dec_res.audit_record.policy_version,
            },
            "scoring": scoring_table,
            "decision": {
                "selected_action": cycle.selected_action.value if cycle.selected_action else None,
                "decision_id": cycle.decision_id,
                "idempotency_key": cycle.idempotency_key,
            },
            "execution": {
                "status": cycle.execution_result.execution_status.value if cycle.execution_result else "N/A",
                "is_duplicate": cycle.execution_result.is_duplicate if cycle.execution_result else False,
                "observation_id": cycle.observation_record.observation_id if cycle.observation_record else None,
                "window_seconds": cycle.observation_record.observation_window_seconds if cycle.observation_record else 0,
            },
            "reconciliation": recon_result,
            "messaging": {
                "delivered_text": cycle.delivered_message.body_text if cycle.delivered_message else None,
                "validation_status": cycle.message_validation.status.value if cycle.message_validation else "N/A",
                "rejection_reasons": [r.value if hasattr(r, "value") else str(r) for r in cycle.message_validation.rejection_reasons] if cycle.message_validation else [],
            },
            "hallucination_guard": hallucination_demo,
            "audit_versions": {
                "policy_version": dec_res.audit_record.policy_version,
                "value_model_version": dec_res.audit_record.value_model_version,
                "transition_model_version": dec_res.audit_record.transition_model_version,
                "feature_schema_version": dec_res.audit_record.feature_schema_version,
            },
        }
        self.send_json(response_payload)

    def send_json(self, payload: dict, status: int = 200):
        data_str = json.dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data_str.encode("utf-8"))


def run_server(port=PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DemoRequestHandler)
    print(f"================================================================")
    print(f"   AI REVENUE RECOVERY DEMO DASHBOARD STARTED")
    print(f"   URL: http://localhost:{port}")
    print(f"   [DEMO / SIMULATED DATA -- CORE ENGINE IS FROZEN]")
    print(f"================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else PORT
    run_server(p)
