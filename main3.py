import os
import random
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any

# --- FIX 1: Streamlit/PyTorch Compatibility ---
# Setting this environment variable prevents Streamlit's file watcher from running
# into issues with PyTorch's internal C++ class paths.
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
# --- END FIX 1 ---

# Networking and LangChain/LangGraph
import requests
import urllib3
import httpx
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ML/DL Libraries
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

DEPARTMENTS = [
    "End-User Support",
    "App Support",
    "Finance App Support L2",
    "Database Operations",
    "Platform Engineering",
    "HR Systems Team",
    "Identity & Access Mgmt"
]

# The default high weight for dependencies when data is insufficient
DEFAULT_HIGH_WEIGHT = 0.9 

# HARDCODED DEPENDENCY_GRAPH (Used as fallback)
DEPENDENCY_GRAPH = {
    "Database Operations": {"Finance App Support L2": 0.8, "End-User Support": 0.3},
    "Platform Engineering": {"End-User Support": 0.5, "App Support": 0.6},
    "Identity & Access Mgmt": {"HR Systems Team": 0.6, "End-User Support": 0.4},
    "Finance App Support L2": {"HR Systems Team": 0.3}
}

# LLM Client Setup
# IMPORTANT: Replace with your actual key or set it as an environment variable


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
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
    dashboard: str  # State key for text summary
    historical_context: Dict[str, str]
    dynamic_dependency_graph: Dict[str, Dict[str, float]] 
    dashboard_df: pd.DataFrame
    total_rev_loss: str

# ------------------------
# GNN CONSTANTS AND MODEL
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
# GNN TRAINING FUNCTIONS
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
    st.info("   -> Preparing historical data for GNN training...")
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
        st.warning("   -> Insufficient data for training. Initializing GNN with random weights.")
        return SimpleGCN(INPUT_FEATURE_SIZE, OUTPUT_FEATURE_SIZE, NUM_DEPARTMENTS, adj_matrix)
        
    model = SimpleGCN(INPUT_FEATURE_SIZE, OUTPUT_FEATURE_SIZE, NUM_DEPARTMENTS, adj_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    NUM_EPOCHS = 30 
    st.info(f"   -> Training GNN for {NUM_EPOCHS} epochs on {len(training_data)} samples...")
    
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
        avg_loss = total_loss / len(training_data)
        
    st.success(f"   -> GNN training complete. Final Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), GNN_WEIGHTS_FILE)
    return model

# ----------------------------------------------------
# NODE: Dynamic Dependency Learning
# ----------------------------------------------------
def learn_dependency_weights(
    tickets: List[Dict[str, Any]],
    departments: List[str],
    causal_window_hours: float = 4.0,
    min_incident_threshold: int = 10,
    default_high_weight: float = DEFAULT_HIGH_WEIGHT
) -> Dict[str, Dict[str, float]]:
    st.info(f"🧠 Starting dynamic dependency learning (Window: {causal_window_hours}h)...")
    
    if not tickets: return {}
    df = pd.DataFrame(tickets)
    df['created_at'] = pd.to_datetime(df['created_at'])
    high_impact_df = df[df['priority'].isin(['1', '2']) | df['impact'].isin(['1', '2'])]
    new_dependency_graph: Dict[str, Dict[str, float]] = {}
    window = pd.Timedelta(hours=causal_window_hours)
    
    for dept_A in departments:
        df_A = df[df['department'] == dept_A]
        total_incidents_A = len(df_A)
        downstream_dependencies: Dict[str, float] = {}
        
        # FALLBACK LOGIC 1: Insufficient data in Dept A
        if total_incidents_A < min_incident_threshold:
            st.warning(f"   -> Insufficient data for **{dept_A}** ({total_incidents_A} tickets). Assigning **default high dependency ({default_high_weight})** to all others.")
            for dept_B in departments:
                if dept_A != dept_B:
                    downstream_dependencies[dept_B] = default_high_weight 
            if downstream_dependencies: new_dependency_graph[dept_A] = downstream_dependencies
            continue
            
        # NORMAL LEARNING
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

    st.success(f"✅ Dynamic dependency learning complete. Found {len(new_dependency_graph)} upstream nodes.")
    return new_dependency_graph

def generate_dynamic_dependency_graph(state: AgentState):
    # This accesses st.session_state, which required the fix above
    dynamic_graph = learn_dependency_weights(
        tickets=state['tickets'],
        departments=DEPARTMENTS,
        causal_window_hours=st.session_state.causal_window, 
        min_incident_threshold=st.session_state.min_threshold,
        default_high_weight=DEFAULT_HIGH_WEIGHT
    )
    # Use dynamically learned graph, but fall back to the hardcoded graph if nothing was learned
    if not dynamic_graph: final_dependency_graph = DEPENDENCY_GRAPH
    else: final_dependency_graph = dynamic_graph
        
    return {"dynamic_dependency_graph": final_dependency_graph}

# ------------------------
# RAG Context Retrieval
# ------------------------
def retrieve_historical_context(department: str, tickets: List[Ticket]) -> str:
    relevant_notes = [
        t['close_notes'] for t in tickets 
        if t['department'] == department and t['resolved'] and t['close_notes'] != "N/A"
    ]
    if not relevant_notes: return "No specific past resolution history (close notes) found."
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
SOP Guidance: {sop_context}

PAST RESOLUTION NOTES (Close Notes):
{combined_notes}
"""
    return summary

# ------------------------
# NODE 1: Data Loader
# ------------------------
@st.cache_data
def load_service_now_tickets(state: AgentState):
    st.info("📥 Fetching latest incidents (or generating fallback data)...")
    
    # --- FALLBACK DATA GENERATION ---
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
            
        if final_dept not in DEPARTMENTS: final_dept = "" 

        revenue_loss = 0.0
        impact_text = t.get("business_impact", "")
        try:
            # Simple extraction of numbers for revenue loss simulation
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
        
    st.success(f"✅ Fetched {len(mapped)} tickets.")
    return {"tickets": mapped, "historical_context": {}, "dynamic_dependency_graph": DEPENDENCY_GRAPH}

# ------------------------
# NODE 2: LLM Classification (Triage Simulation)
# ------------------------
def llm_classify_department(state: AgentState):
    st.info("🧠 Triage: Checking for unassigned tickets...")
    tickets = state["tickets"]
    departments_list = DEPARTMENTS
    UNASSIGNED_CRITERIA = ["", None] 
    classified_count = 0
    
    # NOTE: Skipping actual LLM calls for Streamlit demonstration speed
    for ticket in tickets:
        if ticket["department"] in UNASSIGNED_CRITERIA:
            # Simulate classification
            classified_dept = random.choice(departments_list) 
            ticket["department"] = classified_dept
            classified_count += 1
    
    st.info(f"   -> Simulated classification of {classified_count} unassigned tickets.")
    
    historical_context = {dept: retrieve_historical_context(dept, tickets) for dept in DEPARTMENTS}
    return {"tickets": tickets, "historical_context": historical_context}

# ------------------------
# NODE 3 & 4: Time-Series Prep and LSTM Forecast
# ------------------------
def prepare_ts_data(state: AgentState):
    st.info("⏱ Preparing department-level time-series...")
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
            ts_vector.append([len(day_tickets), day_tickets['revenue_loss'].sum(), (~day_tickets['resolved']).sum()])
        ts_data[dept] = np.array(ts_vector[::-1])
    return {"ts_data": ts_data}

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, num_layers=1, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def forecast_lstm(state: AgentState):
    st.info("📈 Running LSTM forecast for next 15 days...")
    ts_data = state["ts_data"]
    predictions = {}
    scaler = MinMaxScaler()
    model = LSTMModel()
    model.eval()

    for dept, ts in ts_data.items():
        if ts.shape[0] < 15 or ts.sum() == 0:
            predictions[dept] = np.zeros((15, 3))
            continue
        ts_scaled = scaler.fit_transform(ts)
        x_input = torch.tensor(ts_scaled[-15:], dtype=torch.float32).unsqueeze(0)
        forecast = []
        last_input = x_input
        for _ in range(15):
            with torch.no_grad(): out = model(last_input)
            forecast.append(out.numpy().flatten())
            last_input = torch.cat([last_input[:,1:,:], out.unsqueeze(0)], dim=1)
        predictions[dept] = scaler.inverse_transform(np.array(forecast))
    st.success("✅ LSTM forecasting complete.")
    return {"predictions": predictions}

# ------------------------
# NODE 6: Cascading Failure GNN
# ------------------------
def cascading_failure_gnn(state: AgentState):
    st.info("⚡ Running Dynamic GNN cascading failure prediction...")
    lstm_predictions = state["predictions"]
    dependency_graph = state["dynamic_dependency_graph"] 

    adj_matrix = create_adj_matrix(dependency_graph, NUM_DEPARTMENTS)
    gnn_model = SimpleGCN(INPUT_FEATURE_SIZE, OUTPUT_FEATURE_SIZE, NUM_DEPARTMENTS, adj_matrix) 

# [Image of Graph Neural Network Architecture]

    
    if os.path.exists(GNN_WEIGHTS_FILE):
        gnn_model.load_state_dict(torch.load(GNN_WEIGHTS_FILE))
    else:
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
        
    st.success("✅ GNN Prediction complete.")
    return {"predictions": final_probs}

# ------------------------
# NODE 7: RAG Mitigation
# ------------------------
def generate_mitigation(state: AgentState):
    st.info("🛠 Generating RAG-based mitigation steps...")
    mitigation_steps = {}
    historical_context = state["historical_context"]

    # NOTE: Using LLM simulation for speed
    for dept, prob in state["predictions"].items():
        if prob > 0.3:
            context = historical_context.get(dept, "No specific SOP available.")
            
            # Simulated LLM response
            summary = f"Historically, {dept} issues were resolved by checking the user authentication layer (SOP guidance). The average resolution time was 45 minutes."
            action = f"**Actionable Mitigation:** Immediately run a health check on the primary user database connection pool. If risk is > 0.5, schedule a rolling restart of all application servers in region A."
            mitigation_steps[dept] = f"**Resolution Summary:** {summary}\n\n{action}"

        else:
            mitigation_steps[dept] = "Low risk detected. Continue normal monitoring schedule."
            
    st.success("✅ Mitigation steps generated (simulated).")
    return {"mitigation_steps": mitigation_steps}

# ------------------------
# NODE 8: Streamlit Dashboard Display (Renamed to avoid state key conflict)
# ------------------------
def executive_dashboard(state: AgentState):
    st.info("📊 Preparing final Streamlit display data...")
    df = pd.DataFrame(state["tickets"])
    
    total_rev_loss = df['revenue_loss'].sum()
    incidents_data = []

    for dept in DEPARTMENTS:
        risk = state["predictions"].get(dept, 0)
        rev_loss = df[(df['department'] == dept) & (~df['resolved'])]['revenue_loss'].sum()
        steps = state["mitigation_steps"].get(dept, "")
        
        if risk >= 0.7: risk_level_text = "🔴 CRITICAL"
        elif risk >= 0.4: risk_level_text = "🟠 HIGH"
        elif risk >= 0.2: risk_level_text = "🟡 MEDIUM"
        else: risk_level_text = "🟢 LOW"

        incidents_data.append({
            "Department": dept,
            "Risk Probability": f"{risk:.4f}",
            "Risk Level": risk_level_text,
            "Unresolved Revenue Exposure (₹)": f"{rev_loss:,.0f}",
            "Mitigation Steps": steps,
        })
        
    # Store the results DataFrame and total loss in the state
    state['dashboard_df'] = pd.DataFrame(incidents_data)
    state['total_rev_loss'] = f"{total_rev_loss:,.0f}"
    
    st.success("✅ Pipeline execution complete. Displaying results.")
    return state 

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
workflow.add_node("generate_dashboard", executive_dashboard) 

workflow.set_entry_point("data_loader")
workflow.add_edge("data_loader", "llm_classification")
workflow.add_edge("llm_classification", "ts_prep")
workflow.add_edge("ts_prep", "lstm_forecast")
workflow.add_edge("lstm_forecast", "dependency_learning") 
workflow.add_edge("dependency_learning", "cascading_gnn") 
workflow.add_edge("cascading_gnn", "mitigation_rag")
workflow.add_edge("mitigation_rag", "generate_dashboard")
workflow.add_edge("generate_dashboard", END)

app_graph = workflow.compile()

# ------------------------
# STREAMLIT FRONTEND
# ------------------------
st.set_page_config(layout="wide", page_title="Proactive Incident Agent")

# --- FIX 2: UNCONDITIONAL SESSION STATE INITIALIZATION ---
# This block MUST run before any function tries to read these variables
if 'causal_window' not in st.session_state:
    st.session_state.causal_window = 3.0
if 'min_threshold' not in st.session_state:
    st.session_state.min_threshold = 10
if 'run_pipeline' not in st.session_state:
    st.session_state.run_pipeline = False 
# --- END FIX 2 ---

st.title("🚨 Proactive Incident Cascade Predictor")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Graph Parameters")
    # These sliders now safely read and write to the pre-initialized session state
    st.session_state.causal_window = st.slider(
        "Causal Window (Hours)", 
        min_value=1.0, max_value=8.0, value=st.session_state.causal_window, step=0.5,
        help="Max time between incident A (cause) and B (effect)."
    )
    st.session_state.min_threshold = st.slider(
        "Min Incidents (Training)", 
        min_value=5, max_value=30, value=st.session_state.min_threshold, step=5,
        help=f"Min incidents in A required for statistical learning. Below this uses a default weight of {DEFAULT_HIGH_WEIGHT}."
    )
    if st.button("Run Analysis Pipeline"):
        st.session_state.run_pipeline = True
    st.markdown("---")
    st.code("GNN Weights: gnn_model_weights.pth")
    st.code(f"Default High Weight: {DEFAULT_HIGH_WEIGHT}")

if st.button("Clear Cache & Rerun"):
    st.cache_data.clear()
    if os.path.exists(GNN_WEIGHTS_FILE):
        os.remove(GNN_WEIGHTS_FILE)
    st.session_state.run_pipeline = True
    st.rerun()

if st.session_state.get('run_pipeline', False):
    st.subheader("Pipeline Execution Log")
    
    # Run the LangGraph pipeline
    try:
        initial_state = {"tickets": [], "historical_context": {}, "ts_data": {}, "predictions": {}, "mitigation_steps": {}, "dashboard": "", "dynamic_dependency_graph": DEPENDENCY_GRAPH}
        
        # The app.invoke runs the whole process
        final_result = app_graph.invoke(initial_state)
        st.session_state.final_result = final_result
        st.session_state.run_pipeline = False
        st.rerun() 
        
    except Exception as e:
        st.error(f"An error occurred during pipeline execution: {e}")
        st.session_state.run_pipeline = False
        st.session_state.error = str(e)


if 'final_result' in st.session_state:
    result = st.session_state.final_result
    
    st.header("📊 Final Executive Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Revenue Exposure (30 Days)",
            value=f"₹{result.get('total_rev_loss', 'N/A')}",
            delta="Simulated Data"
        )
    with col2:
        st.metric(
            label="Upstream Nodes Learned",
            value=len(result.get('dynamic_dependency_graph', {})),
            delta="Dynamic Weights Used"
        )
    with col3:
        # Display the GNN prediction results in a clearer table
        st.markdown("**Incident Risk Forecast (GNN Output)**")
        st.dataframe(
            result['dashboard_df'][['Department', 'Risk Level', 'Risk Probability']],
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    st.header("⚡ Department-Specific Proactive Steps")
    
    # Display detailed mitigation steps in expanders
    for _, row in result['dashboard_df'].iterrows():
        dept = row['Department']
        risk_level = row['Risk Level']
        
        with st.expander(f"{risk_level}: **{dept}** - Predicted Risk: {row['Risk Probability']}"):
            st.markdown(f"**Unresolved Revenue Exposure:** ₹{row['Unresolved Revenue Exposure (₹)']}")
            st.markdown(row['Mitigation Steps'])
            
            # Show the underlying RAG context for transparency
            with st.expander("Show Historical RAG Context"):
                 st.code(result['historical_context'].get(dept, "Context not found."), language='markdown')

    st.markdown("---")
    st.header("🔗 Dynamic Dependency Graph (Adjacency Matrix)")
    
    # Display the learned graph structure
    st.json(result['dynamic_dependency_graph'])


if not st.session_state.get('run_pipeline', False) and 'final_result' not in st.session_state:
    st.warning("Click 'Run Analysis Pipeline' in the sidebar to start the process.")

# --- End of Streamlit Code ---