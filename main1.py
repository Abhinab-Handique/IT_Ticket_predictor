import os
import random
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any
import requests
import urllib3
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from sklearn.preprocessing import MinMaxScaler

# Silence InsecureRequestWarning for ServiceNow self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------
# CONFIG & SETUP
# ------------------------
SERVICENOW_INSTANCE = ""
USERNAME = "admin"
PASSWORD = ""
GNN_WEIGHTS_FILE = "gnn_model_weights.pth"
LSTM_WEIGHTS_FILE = "lstm_model_weights.pth"
LSTM_SEQUENCE_LENGTH = 15

DEPARTMENTS = [
    "End-User Support",
    "App Support",
    "Finance App Support L2",
    "Database Operations",
    "Platform Engineering",
    "HR Systems Team",
    "Identity & Access Mgmt"
]

# HARDCODED DEPENDENCY_GRAPH (Used as fallback)
DEPENDENCY_GRAPH = {
    "Database Operations": {"Finance App Support L2": 0.8, "End-User Support": 0.3},
    "Platform Engineering": {"End-User Support": 0.5, "App Support": 0.6},
    "Identity & Access Mgmt": {"HR Systems Team": 0.6, "End-User Support": 0.4},
    "Finance App Support L2": {"HR Systems Team": 0.3}
}

# LLM Client Setup


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
)

# ------------------------
# DATA TYPES
# ------------------------
class Ticket(TypedDict):
    id: str
    department: str
    severity: str
    business_impact: str
    resolved: bool
    created_at: str
    description: str
    revenue_loss: float
    close_notes: str 
    priority: str
    impact: str

class AgentState(TypedDict):
    tickets: List[Ticket]
    ts_data: Dict[str, Any]
    predictions: Dict[str, Any]
    mitigation_steps: Dict[str, str]
    # Dashboard now holds the final markdown string
    dashboard: str 
    historical_context: Dict[str, str]
    dynamic_dependency_graph: Dict[str, Dict[str, float]] 

# ------------------------
# GNN CONSTANTS AND MODEL DEFINITION
# ------------------------
DEPT_TO_IDX = {dept: i for i, dept in enumerate(DEPARTMENTS)}
IDX_TO_DEPT = {i: dept for i, dept in enumerate(DEPARTMENTS)}
NUM_DEPARTMENTS = len(DEPARTMENTS)
INPUT_FEATURE_SIZE = 1 
OUTPUT_FEATURE_SIZE = 1 

class SimpleGCN(nn.Module):
    def __init__(self, input_size, output_size, num_nodes, adj_matrix):
        super(SimpleGCN, self).__init__()
        self.W = nn.Linear(input_size, output_size)
        
        A = torch.tensor(adj_matrix, dtype=torch.float32)
        I = torch.eye(num_nodes)
        A_hat = A + I
        D = torch.sum(A_hat, dim=1)
        D[D == 0] = 1e-6 
        D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
        
        self.A_norm = torch.matmul(torch.matmul(D_inv_sqrt, A_hat), D_inv_sqrt)
        
    def forward(self, H):
        H = self.W(H)
        H = torch.matmul(self.A_norm, H)
        H = torch.sigmoid(H)
        return H

def create_adj_matrix(dependency_graph: Dict[str, Dict[str, float]], num_nodes: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes))
    for upstream, downstream_deps in dependency_graph.items():
        if upstream in DEPT_TO_IDX:
            u_idx = DEPT_TO_IDX[upstream]
            for downstream, weight in downstream_deps.items():
                if downstream in DEPT_TO_IDX:
                    d_idx = DEPT_TO_IDX[downstream]
                    adj[u_idx, d_idx] = weight
    return adj

# ------------------------
# GNN & LSTM TRAINING FUNCTIONS
# ------------------------
def engineer_cascade_label(ticket: dict) -> float:
    impact_map = {"1": 1.0, "2": 0.6, "3": 0.2}
    priority_map = {"1": 1.0, "2": 0.7, "3": 0.4, "4": 0.1}
    impact_score = impact_map.get(ticket.get("impact", "3"), 0.2)
    priority_score = priority_map.get(ticket.get("priority", "4"), 0.1)
    combined_score = (impact_score * 0.6) + (priority_score * 0.4)
    if "multiple customers" in ticket.get("description", "").lower():
        combined_score = min(1.0, combined_score + 0.1)
    return combined_score

def prepare_gnn_training_data(tickets: List[Ticket]):
    print("   -> Preparing historical data for GNN training...")
    df = pd.DataFrame(tickets)
    df['created_at'] = pd.to_datetime(df['created_at'])
    training_samples = []
    
    for day_offset in range(1, 31): 
        target_date = datetime.now() - timedelta(days=day_offset)
        daily_df = df[df['created_at'].dt.date == target_date.date()]
        if daily_df.empty: continue
        H_day = np.zeros((NUM_DEPARTMENTS, INPUT_FEATURE_SIZE))
        Y_day = np.zeros(NUM_DEPARTMENTS)
        for dept_idx, dept in IDX_TO_DEPT.items():
            dept_tickets = daily_df[daily_df['department'] == dept]
            H_day[dept_idx, 0] = len(dept_tickets) / 5.0 
            if not dept_tickets.empty:
                Y_day[dept_idx] = dept_tickets.apply(engineer_cascade_label, axis=1).max()
        training_samples.append({
            'H': torch.tensor(H_day, dtype=torch.float32),
            'Y': torch.tensor(Y_day, dtype=torch.float32)
        })
    return training_samples

def train_gnn_model(training_data, adj_matrix):
    if not training_data:
        return SimpleGCN(INPUT_FEATURE_SIZE, OUTPUT_FEATURE_SIZE, NUM_DEPARTMENTS, adj_matrix)
        
    model = SimpleGCN(INPUT_FEATURE_SIZE, OUTPUT_FEATURE_SIZE, NUM_DEPARTMENTS, adj_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    NUM_EPOCHS = 30 
    
    print(f"   -> Training GNN for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        model.train()
        for sample in training_data:
            H = sample['H']
            Y_true = sample['Y'].unsqueeze(1) 
            optimizer.zero_grad()
            Y_pred = model(H)
            loss = criterion(Y_pred, Y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    torch.save(model.state_dict(), GNN_WEIGHTS_FILE)
    print(f"   -> GNN training complete. Weights saved to {GNN_WEIGHTS_FILE}")
    return model

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, num_layers=1, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(ts_data: Dict[str, np.ndarray]):
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    NUM_EPOCHS = 5

    all_ts_data = []
    for dept, ts in ts_data.items():
        if ts.shape[0] >= LSTM_SEQUENCE_LENGTH + 1:
            for i in range(ts.shape[0] - LSTM_SEQUENCE_LENGTH):
                X = ts[i:i + LSTM_SEQUENCE_LENGTH]
                Y = ts[i + LSTM_SEQUENCE_LENGTH]
                all_ts_data.append((X, Y))
    
    if not all_ts_data:
        print("   -> Insufficient data for LSTM training. Using random weights.")
        return model

    X_train = torch.tensor(np.array([item[0] for item in all_ts_data]), dtype=torch.float32)
    Y_train = torch.tensor(np.array([item[1] for item in all_ts_data]), dtype=torch.float32)

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), LSTM_WEIGHTS_FILE)
    print("   -> LSTM model trained and weights saved.")
    return model

def get_trained_lstm_model(ts_data: Dict[str, np.ndarray]):
    if os.path.exists(LSTM_WEIGHTS_FILE):
        model = LSTMModel()
        try:
            model.load_state_dict(torch.load(LSTM_WEIGHTS_FILE))
            return model
        except Exception as e:
            print(f"Failed to load LSTM weights: {e}. Retraining.")
            return train_lstm_model(ts_data)
    else:
        return train_lstm_model(ts_data)

# ------------------------
# RAG Context Retrieval
# ------------------------
def retrieve_historical_context(department: str, tickets: List[Ticket]) -> str:
    relevant_notes = [
        t['close_notes'] for t in tickets 
        if t['department'] == department and t['resolved'] and t['close_notes'] != "N/A"
    ]
    
    if not relevant_notes:
        return "No specific past resolution history (close notes) found."

    combined_notes = "\n---\n".join(relevant_notes)
    MAX_CONTEXT_LENGTH = 1500
    if len(combined_notes) > MAX_CONTEXT_LENGTH:
        combined_notes = combined_notes[:MAX_CONTEXT_LENGTH] + "\n... (Context truncated)"
    
    sop_context = {
        "Database Operations": "Past resolution for database-related incidents often involves optimizing query indexes and ensuring replication status is green.",
        "Finance App Support L2": "Standard Operating Procedure (SOP) 4.1 mandates checking finance ledger integrity before any data rollback.",
        "End-User Support": "Most cascading failures stem from identity access management issues. Ensure users can log in before escalating L1 issues."
    }.get(department, "")
    
    summary = f"""
--- START HISTORICAL CONTEXT FOR {department} ---
SOP Guidance: {sop_context}

PAST RESOLUTION NOTES (Close Notes):
{combined_notes}
--- END HISTORICAL CONTEXT ---
"""
    return summary

# ----------------------------------------------------
# 🆕 CAUSAL ANALYSIS HELPER FUNCTION
# ----------------------------------------------------

def identify_core_trigger(department: str, tickets: List[Ticket]) -> str:
    """
    Identifies the most critical unresolved incident or pattern in a department 
    to serve as the root cause trigger for the dashboard.
    """
    unresolved_df = pd.DataFrame([t for t in tickets if not t['resolved'] and t['department'] == department])
    if unresolved_df.empty:
        return "No specific unresolved tickets. Risk is based on historical trends."
    
    # Sort by Priority (1=Highest) then Impact (1=Highest)
    unresolved_df['P_int'] = unresolved_df['priority'].astype(int)
    unresolved_df['I_int'] = unresolved_df['impact'].astype(int)
    
    # Select the single most critical ticket
    most_critical = unresolved_df.sort_values(by=['P_int', 'I_int'], ascending=[True, True]).iloc[0]
    
    trigger_desc = f"""
    **Highest Priority Unresolved Incident (Root Trigger):** {most_critical['id']} - "{most_critical['description']}" 
    (Priority:{most_critical['priority']}, Impact:{most_critical['impact']}).
    """
    return trigger_desc

# ------------------------
# NODE 1: Data Loader
# ------------------------
def load_service_now_tickets(state: AgentState):
    print("📥 Fetching latest incidents from ServiceNow...")
    
    params = {
        "sysparm_limit": 200,
        "sysparm_query": "ORDERBYDESCsys_created_on",
        "sysparm_fields": "number,short_description,description,priority,impact,state,sys_created_on,close_notes,assignment_group,business_impact" 
    }
    try:
        response = requests.get(
            f"{SERVICENOW_INSTANCE}/api/now/table/incident",
            auth=(USERNAME, PASSWORD),
            headers={"Accept": "application/json"},
            params=params,
            verify=False,
            timeout=10
        )
        response.raise_for_status()
        raw_tickets = response.json().get("result", [])
    except Exception as e:
        print(f"❌ Error fetching tickets: {e}. Generating fallback data.")
        
        # Fallback data generation
        raw_tickets = []
        for i in range(2000):
            random_dept_name = random.choice(DEPARTMENTS)
            random_day = datetime.now().date() - timedelta(days=random.randint(1, 30))
            created_datetime = datetime.combine(random_day, datetime.min.time()) + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))

            raw_tickets.append({
                "number": f"INC{i}", "short_description": f"Failure {i} - DB query slowdown", 
                "description": f"Detailed failure description for ticket {i}.", 
                "priority": str(random.randint(1,4)), "impact": str(random.randint(1,3)), 
                "state": random.choice(["6", "1"]), 
                "sys_created_on": created_datetime.isoformat(), 
                "close_notes": "Database connection pool exhausted. Increased max connections to 500." if i % 5 == 0 else "User cache cleared.",
                "business_impact": f"Low - downtime of 30 minutes, revenue loss ₹{random.randint(1000, 50000)}.",
                "assignment_group": random.choice(["", {"display_value": random_dept_name}]) if i % 3 != 0 else ""
            })

    mapped: List[Ticket] = []
    severity_map = {"1": "Critical", "2": "High", "3": "Medium", "4": "Low"}

    for t in raw_tickets:
        assigned_group = t.get("assignment_group", "")
        final_dept = ""
        if isinstance(assigned_group, dict):
            final_dept = assigned_group.get('display_value', "")
        elif isinstance(assigned_group, str):
            final_dept = assigned_group
            
        if final_dept not in DEPARTMENTS:
            final_dept = "" 

        revenue_loss = 0.0
        impact_text = t.get("business_impact", "")
        try:
            revenue_loss = float(int(''.join(filter(str.isdigit, impact_text)))) if any(c.isdigit() for c in impact_text) else random.uniform(10000, 500000)
        except:
            revenue_loss = random.uniform(10000, 500000)

        mapped.append({
            "id": t.get("number", "N/A"),
            "department": final_dept, 
            "severity": severity_map.get(t.get("priority", "4"), "Low"),
            "business_impact": t.get("business_impact", "N/A"),
            "resolved": t.get("state") in ["6", "7"],
            "created_at": t.get("sys_created_on", datetime.now().isoformat()),
            "description": t.get("short_description", "No description"),
            "revenue_loss": revenue_loss,
            "close_notes": t.get("close_notes", "N/A"),
            "priority": t.get("priority", "4"),
            "impact": t.get("impact", "3"),
        })
        
    print(f"✅ Fetched {len(mapped)} tickets.")
    return {"tickets": mapped, "historical_context": {}, "dynamic_dependency_graph": DEPENDENCY_GRAPH}

# ------------------------
# NODE 2: LLM Classification (Conditional Triage)
# ------------------------
def llm_classify_department(state: AgentState):
    print("🧠 Triage: Checking for unassigned tickets...")
    tickets = state["tickets"]
    departments_list = DEPARTMENTS
    UNASSIGNED_CRITERIA = ["", None] 
    classified_count = 0
    
    for ticket in tickets:
        current_dept = ticket["department"]
        
        if current_dept in UNASSIGNED_CRITERIA:
            # Simulating LLM call for triage
            classified_dept = random.choice(departments_list) 
            ticket["department"] = classified_dept
            classified_count += 1
    
    if classified_count == 0:
        print("✅ All tickets were already assigned or could not be classified.")
    
    historical_context = {dept: retrieve_historical_context(dept, tickets) for dept in DEPARTMENTS}
    
    return {"tickets": tickets, "historical_context": historical_context}


# ------------------------
# NODE 3: Prepare Time-Series Data
# ------------------------
def prepare_ts_data(state: AgentState):
    print("⏱ Preparing department-level time-series...")
    tickets = state["tickets"]
    df = pd.DataFrame(tickets)
    df['created_at'] = pd.to_datetime(df['created_at'])
    ts_data = {}

    for dept in DEPARTMENTS:
        dept_df = df[df['department'] == dept].sort_values('created_at')
        ts_vector = []
        for day_offset in range(30):
            date_filter = datetime.now() - timedelta(days=day_offset)
            day_tickets = dept_df[dept_df['created_at'].dt.date == date_filter.date()]
            ts_vector.append([
                len(day_tickets),                          
                day_tickets['revenue_loss'].sum(),         
                (~day_tickets['resolved']).sum()           
            ])
        ts_data[dept] = np.array(ts_vector[::-1])
    return {"ts_data": ts_data}

# ------------------------
# NODE 4: LSTM Forecast
# ------------------------
def forecast_lstm(state: AgentState):
    print("📈 Running LSTM forecast for next 15 days...")
    ts_data = state["ts_data"]
    predictions = {}
    scaler = MinMaxScaler()

    model = get_trained_lstm_model(ts_data)
    model.eval()

    for dept, ts in ts_data.items():
        if ts.shape[0] < LSTM_SEQUENCE_LENGTH or ts.sum() == 0:
            predictions[dept] = np.zeros((15, 3))
            continue
            
        ts_scaled = scaler.fit_transform(ts)
        x_input = torch.tensor(ts_scaled[-LSTM_SEQUENCE_LENGTH:], dtype=torch.float32).unsqueeze(0)
        forecast = []
        last_input = x_input
        for _ in range(15):
            with torch.no_grad():
                out = model(last_input)
            forecast.append(out.numpy().flatten())
            last_input = torch.cat([last_input[:,1:,:], out.unsqueeze(0)], dim=1)
            
        predictions[dept] = scaler.inverse_transform(np.array(forecast))
        
    return {"predictions": predictions}

# ----------------------------------------------------
# NODE 5: Dynamic Dependency Learning
# ----------------------------------------------------
def learn_dependency_weights(
    tickets: List[Dict[str, Any]],
    departments: List[str],
    causal_window_hours: float = 4.0,
    min_incident_threshold: int = 10
) -> Dict[str, Dict[str, float]]:
    print(f"\n🧠 Starting dynamic dependency learning (Causal Window: {causal_window_hours}h)...")
    if not tickets: return {}
    df = pd.DataFrame(tickets)
    df['created_at'] = pd.to_datetime(df['created_at'])
    high_impact_df = df[df['priority'].isin(['1', '2']) | df['impact'].isin(['1', '2'])]
    new_dependency_graph: Dict[str, Dict[str, float]] = {}
    window = pd.Timedelta(hours=causal_window_hours)
    
    for dept_A in departments:
        df_A = df[df['department'] == dept_A]
        total_incidents_A = len(df_A)
        if total_incidents_A < min_incident_threshold: continue 

        downstream_dependencies: Dict[str, float] = {}
        for dept_B in departments:
            if dept_A == dept_B: continue 
            df_B_high_impact = high_impact_df[high_impact_df['department'] == dept_B]
            sequence_count = 0
            
            for _, row_A in df_A.iterrows():
                time_A = row_A['created_at']
                downstream_events = df_B_high_impact[
                    (df_B_high_impact['created_at'] > time_A) &
                    (df_B_high_impact['created_at'] <= time_A + window)
                ]
                if not downstream_events.empty: sequence_count += 1
            
            if sequence_count > 0:
                weight = sequence_count / total_incidents_A
                downstream_dependencies[dept_B] = round(min(max(weight, 0.1), 1.0), 4)

        if downstream_dependencies: new_dependency_graph[dept_A] = downstream_dependencies

    print(f"✅ Dynamic dependency learning complete. Found {len(new_dependency_graph)} upstream nodes.")
    return new_dependency_graph

def generate_dynamic_dependency_graph(state: AgentState):
    dynamic_graph = learn_dependency_weights(
        tickets=state['tickets'],
        departments=DEPARTMENTS,
        causal_window_hours=3.0, 
        min_incident_threshold=10
    )
    
    if not dynamic_graph:
        print("   -> Dynamic graph empty or unreliable. Falling back to static DEPENDENCY_GRAPH.")
        final_dependency_graph = DEPENDENCY_GRAPH
    else:
        print("   -> Using dynamically learned dependency graph.")
        final_dependency_graph = dynamic_graph
        
    return {"dynamic_dependency_graph": final_dependency_graph}


# ------------------------
# NODE 6: Cascading Failure GNN (Dynamic Weights)
# ------------------------
def cascading_failure_gnn(state: AgentState):
    print("⚡ Running Dynamic GNN cascading failure prediction...")
    lstm_predictions = state["predictions"]
    dependency_graph = state["dynamic_dependency_graph"] 
    adj_matrix = create_adj_matrix(dependency_graph, NUM_DEPARTMENTS)
    
    gnn_model = SimpleGCN(
        input_size=INPUT_FEATURE_SIZE,
        output_size=OUTPUT_FEATURE_SIZE,
        num_nodes=NUM_DEPARTMENTS,
        adj_matrix=adj_matrix
    )
    
    if os.path.exists(GNN_WEIGHTS_FILE):
        gnn_model.load_state_dict(torch.load(GNN_WEIGHTS_FILE))
    else:
        print("   -> No trained weights found. Initiating GNN training on historical data.")
        training_data = prepare_gnn_training_data(state['tickets'])
        gnn_model = train_gnn_model(training_data, adj_matrix) 
    
    node_features = np.zeros((NUM_DEPARTMENTS, INPUT_FEATURE_SIZE))
    for dept, idx in DEPT_TO_IDX.items():
        avg_incidents = np.mean(lstm_predictions.get(dept, np.zeros((15, 3)))[:, 0])
        base_risk_score = min(1.0, avg_incidents / 5.0) 
        node_features[idx, 0] = base_risk_score

    H_initial = torch.tensor(node_features, dtype=torch.float32)

    gnn_model.eval() 
    with torch.no_grad():
        H_final = gnn_model(H_initial)
        
    final_probs = {}
    for idx in range(NUM_DEPARTMENTS):
        dept = IDX_TO_DEPT[idx]
        final_probs[dept] = H_final[idx].item()
        
    return {"predictions": final_probs}

# ------------------------
# NODE 7: RAG Mitigation (Updated for Causal Chain)
# ------------------------
def generate_mitigation(state: AgentState):
    print("🛠 Generating RAG-based mitigation steps...")
    mitigation_steps = {}
    historical_context = state["historical_context"]
    dynamic_graph = state["dynamic_dependency_graph"] 

    for dept, prob in state["predictions"].items():
        if prob > 0.3:
            context = historical_context.get(dept, "No specific SOP available.")
            
            # 🆕 Identify the root trigger for this department
            trigger = identify_core_trigger(dept, state['tickets'])
            
            # 🆕 Get departments that depend on this one
            downstream = dynamic_graph.get(dept, {})
            downstream_list = ", ".join([f"{d} (Weight: {w:.2f})" for d, w in downstream.items()]) or "None"
            
            prompt = f"""
**RAG CONTEXT (Past Incidents/SOPs/Resolutions):**
{context}
---
**CAUSAL ANALYSIS & RISK ASSESSMENT:**
Department (Upstream Risk): {dept}
Predicted Cascading Failure Risk (0-1): {prob:.2f}
{trigger}
**Downstream Dependent Departments (Weight):** {downstream_list}

Based on the RAG CONTEXT, the **ROOT TRIGGER**, and the **DOWNSTREAM DEPENDENCIES**, provide:
1. **Critical Analysis:** How the Root Trigger incident in **{dept}** relates to and is likely to cause failure in the listed Downstream Departments.
2. **Actionable Mitigation:** Detailed, proactive steps for the {dept} team to resolve the ROOT TRIGGER and prevent the cascade.
"""
            try:
                # Simulating LLM call for mitigation
                if dept == "Database Operations":
                    mitigation_text = f"**Critical Analysis:** The current DB connection pool issue is known to immediately impact Finance App Support L2 (weight 0.8) due to shared infrastructure. The cascade risk is very high.\n\n**Actionable Mitigation:** Immediately run a health check on the primary user database connection pool. If risk is > 0.5, schedule a rolling restart of all database servers, prioritizing resolution of the root trigger {trigger.split('-')[0].strip()}."
                else:
                    mitigation_text = f"**Critical Analysis:** Past risks were mitigated by simple service restarts. The dependency on {downstream_list} suggests a shared resource issue, likely authentication or platform.\n\n**Actionable Mitigation:** Verify system logs for high CPU or memory usage. Prepare to escalate to L3 if incidents double within 2 hours, focusing efforts on the core issue: {trigger.split('-')[1].strip()}"

                mitigation_steps[dept] = mitigation_text
            except Exception as e:
                print(f"LLM call failed for {dept}: {e}")
                mitigation_steps[dept] = "Fallback: monitor logs, ensure backups are verified, and restart impacted services if risk is high."
        else:
            mitigation_steps[dept] = "Low risk detected. Continue normal monitoring schedule."
            
    return {"mitigation_steps": mitigation_steps}

# ------------------------
# NODE 8: Executive Dashboard (Updated for Causal Chain)
# ------------------------
def executive_dashboard(state: AgentState):
    print("📊 Preparing executive dashboard with Causal Analysis...")
    df = pd.DataFrame(state["tickets"])
    lines = []
    
    total_rev_loss = df['revenue_loss'].sum()
    lines.append("## 🚨 Proactive Causal Incident Forecast Summary\n")
    lines.append(f"**Total Revenue Exposure (Last 30 Days):** ₹{total_rev_loss:,.0f}\n")
    lines.append("---")
    
    dynamic_graph = state["dynamic_dependency_graph"]
    
    for dept in DEPARTMENTS:
        risk = state["predictions"].get(dept, 0)
        rev_loss = df[(df['department'] == dept) & (~df['resolved'])]['revenue_loss'].sum()
        steps = state["mitigation_steps"].get(dept, "")
        
        # 🆕 Identify the root cause for display
        trigger = identify_core_trigger(dept, state['tickets'])
        
        # 🆕 List downstream dependencies
        downstream = dynamic_graph.get(dept, {})
        downstream_list = "\n".join([f"    - {d} (Weight: {w:.2f})" for d, w in downstream.items()])
        downstream_section = f"""
* **Downstream Cascades Predicted:**
{downstream_list if downstream_list else "    - None (Isolated Risk)"}
"""
        
        risk_level = "🔴 CRITICAL" if risk >= 0.7 else ("🟠 HIGH" if risk >= 0.4 else "🟡 MEDIUM" if risk >= 0.2 else "🟢 LOW")
        
        lines.append(f"""
### {risk_level}: {dept}
* **Predicted Cascade Probability:** **{risk:.2f}**
* **Current Unresolved Revenue Exposure:** ₹{rev_loss:,.0f}
{downstream_section}
* **Core Internal Trigger (Highest Risk Ticket):**
> {trigger.strip()}

* **Causal Analysis & Proactive Mitigation:**
> {steps.replace('\n', '\n> ')}
        """)
        
    dashboard_text = "\n".join(lines)
    return {"dashboard": dashboard_text}

# ------------------------
# BUILD LANGGRAPH WORKFLOW
# ------------------------
workflow = StateGraph(AgentState)
workflow.add_node("data_loader", load_service_now_tickets)
workflow.add_node("llm_classification", llm_classify_department)
workflow.add_node("ts_prep", prepare_ts_data)
workflow.add_node("lstm_forecast", forecast_lstm)
workflow.add_node("dependency_learning", generate_dynamic_dependency_graph) 
workflow.add_node("cascading_gnn", cascading_failure_gnn)
workflow.add_node("mitigation_rag", generate_mitigation)
workflow.add_node("dashboard", executive_dashboard)

workflow.set_entry_point("data_loader")
workflow.add_edge("data_loader", "llm_classification")
workflow.add_edge("llm_classification", "ts_prep")
workflow.add_edge("ts_prep", "lstm_forecast")
workflow.add_edge("lstm_forecast", "dependency_learning") 
workflow.add_edge("dependency_learning", "cascading_gnn") 
workflow.add_edge("cascading_gnn", "mitigation_rag")
workflow.add_edge("mitigation_rag", "dashboard")
workflow.add_edge("dashboard", END)

APP_GRAPH = workflow.compile()

# ------------------------
# EXECUTION
# ------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PROACTIVE INCIDENT AGENT (DYNAMIC DEPENDENCY + CAUSAL ANALYSIS) STARTED")
    print("="*70 + "\n")
    initial_state = {"tickets": [], "historical_context": {}, "ts_data": {}, "predictions": {}, "mitigation_steps": {}, "dashboard": "", "dynamic_dependency_graph": DEPENDENCY_GRAPH}
    
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "dummy-key-for-groq":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!! WARNING: GROQ_API_KEY is missing or set to a dummy value.!!")
        print("!! LLM steps (triage, mitigation) will use fallback text.   !!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    try:
        result = APP_GRAPH.invoke(initial_state)
        print("\n" + "="*70)
        print("FINAL EXECUTIVE DASHBOARD (CAUSAL ANALYSIS)")
        print("="*70 + "\n")
        print(result["dashboard"])
        print("\n" + "="*70)
        print("DYNAMIC DEPENDENCY GRAPH (Learned Structure)")
        print("="*70)
        print(json.dumps(result["dynamic_dependency_graph"], indent=4))
        
    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        print("Please ensure all dependencies are installed.")