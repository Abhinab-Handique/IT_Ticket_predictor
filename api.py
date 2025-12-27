# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_core import APP_GRAPH, AgentState, DEPENDENCY_GRAPH
import json
import os

app = FastAPI(title="Incident Predictor API")

class RunPipelineRequest(BaseModel):
    causal_window: float = 3.0
    min_threshold: int = 100

@app.post("/run_pipeline")
async def run_pipeline(request: RunPipelineRequest):
    """
    Triggers the LangGraph pipeline with dynamic parameters.
    """
    try:
        # Initialize state with request parameters
        initial_state: AgentState = {
            "tickets": [],
            "historical_context": {},
            "ts_data": {},
            "predictions": {},
            "mitigation_steps": {},
            "dashboard": "",
            "dynamic_dependency_graph": DEPENDENCY_GRAPH,
            "causal_window": request.causal_window,
            "min_threshold": request.min_threshold
        }
        
        # Invoke the LangGraph pipeline
        result = APP_GRAPH.invoke(initial_state)
        
        # The 'dashboard' key contains the structured JSON output for the frontend
        return result['dashboard']
    
    except Exception as e:
        return {"error": str(e), "message": "Pipeline execution failed."}

@app.get("/status")
def get_status():
    return {"status": "ok", "message": "Incident Predictor API is running."}

if __name__ == "__main__":
    import uvicorn
    # Make sure this runs on a different port than Flask (e.g., 8000)
    # Use: uvicorn api:app --reload --port 8000
    print("Run using: uvicorn api:app --reload --port 8000")
    # uvicorn.run(app, host="0.0.0.0", port=8000)