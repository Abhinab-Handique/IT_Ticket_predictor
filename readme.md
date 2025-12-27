

# ⚡ Proactive IT Incident Agent: AI-Driven Risk Mitigation

The **Proactive IT Incident Agent** is an advanced operational intelligence system designed to shift IT support from a **reactive** (fixing what is broken) to a **proactive** (preventing what will break) posture.

By combining **Time-Series Forecasting (LSTM)**, **Graph Neural Networks (GNN)**, and **Large Language Models (LLM)**, the system predicts future incident trends, identifies causal "domino effects" between departments, and prioritizes resolution based on predicted business impact.

---

## 🏗️ Project Overview

In traditional IT environments, teams respond to incidents as they appear. This project uses **LangGraph** to orchestrate a sophisticated workflow that treats incidents as parts of an interconnected system. It identifies "Root Triggers" that, if left unresolved, will cause cascading failures across downstream services.

### The Logic Flow:

1. **Data Ingestion:** Real-time ticket fetching from ServiceNow.
2. **Forecasting:** Predicting future workload stress using LSTM.
3. **Graph Mapping:** Building a dynamic dependency map of IT departments.
4. **Cascade Prediction:** Using a GNN to calculate the probability of a "Domino Effect."
5. **Scenario Prediction:** LLM-driven prediction of the *next* likely failure ticket.
6. **Prioritization:** Ranking departments by a business-weighted **P-Score**.

---

## 🧩 Core Components

### 1. Trend Forecasting (LSTM)

The system employs a **Long Short-Term Memory (LSTM)** network to analyze historical incident volume and unresolved backlogs. It predicts the next 15 days of "stress" for each department.

```python
# Sequential pattern learning for future load
class IncidentLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(IncidentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3) # Predicts Count, Loss, and Backlog

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

```

### 2. Cascading Failure Prediction (GNN)

To model systemic risk, we use a **Graph Convolutional Network (GCN)**. It treats departments as nodes and their dependencies as edges, allowing risk signals to propagate through the network.

```python
# Propagating risk across the dependency graph
class CascadeGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CascadeGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return torch.sigmoid(self.conv2(x, edge_index))

```

### 3. Potential Issue Prediction (LLM + RAG)

Instead of generic alerts, we use **Retrieval-Augmented Generation (RAG)** to provide a narrative. The LLM looks at the current highest-risk ticket and predicts the *specific* next issue that will likely break downstream.

```python
# Predicting the next specific ticket scenario
prompt = f"""
Current Root Trigger: {trigger}
Downstream Dependencies: {downstream_list}
Analysis: Based on historical context, predict the NEXT potential issue 
that will occur if the trigger is NOT resolved within the current TTR.
"""

```

---

## 📊 The P-Score: Strategic Prioritization

The final output is the **Prioritization Score (P-Score)**, a 0-100 metric that tells the operations team exactly where to focus. It is calculated using the following formula:

* **R (Cascade Risk):** 45% weight – How likely is this to break other things?
* **L (Projected Loss):** 40% weight – What is the financial cost of delaying the fix?
* **I (Downstream Impact):** 15% weight – How many other teams rely on this service?

---

## 🖥️ Operational Dashboard

The integrated **Flask Dashboard** visualizes these complex metrics for easy consumption.

* **15-Day Forecasts:** Visualizing historical vs. predicted ticket volume.
* **Causal Analysis:** Detailed warnings about what will break next.
* **Ranked Priority Table:** A clear "To-Do" list for senior engineering staff.

---

## 🚀 Getting Started

### 🔧 Installation

1. **Clone the Repo:**
```bash
git clone https://github.com/your-username/proactive-incident-agent.git
cd proactive-incident-agent

```


2. **Install Dependencies:**
```bash
pip install torch torch-geometric flask langgraph pandas numpy

```


3. **Run the Agent & Dashboard:**
```bash
python app.py

```



### 📈 Usage

* **Initial Run:** The system will train the LSTM and GNN models on your ServiceNow data and save the weights (`.pth` files).
* **Daily Monitoring:** Access the dashboard at `http://localhost:5000` to see the updated priority list and potential failure scenarios.

---

## 💡 Why This Project?

Standard ITSM tools are history books; they tell you what happened. This agent is a **weather forecast** for your IT infrastructure, allowing you to stop the rain before the storm starts.

---

Would you like me to refine any of the code snippets or add a section on how to connect your specific ServiceNow instance?