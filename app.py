# app.py

from flask import Flask, render_template_string, request, jsonify
import requests
import json
import markdown

app = Flask(__name__)

# URL for the FastAPI backend
API_URL = "http://127.0.0.1:8000/run_pipeline"

# Simple HTML template for the Flask frontend
HTML_TEMPLATE = """
<!doctype html>
<title>Proactive Incident Predictor</title>
<style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    .container { max-width: 1200px; margin: auto; }
    .sidebar { float: left; width: 250px; padding: 20px; background: #f9f9f9; border-radius: 8px; margin-right: 20px; }
    .main-content { margin-left: 270px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
    h1 { color: #d9534f; }
    .result { margin-top: 20px; }
    .error { color: red; font-weight: bold; }
    .risk-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    .risk-table th, .risk-table td { border: 1px solid #ccc; padding: 8px; text-align: left; }
    .risk-table th { background-color: #eee; }
    .CRITICAL { color: white; background-color: #d9534f; }
    .HIGH { color: black; background-color: #f0ad4e; }
    .MEDIUM { color: black; background-color: #5cb85c; }
    .LOW { color: white; background-color: #5bc0de; }
</style>
<div class="container">
    <h1>🚨 Proactive Incident Cascade Predictor</h1>

    <div class="sidebar">
        <h2>Parameters</h2>
        <form method="POST" action="/run">
            <label for="causal_window">Causal Window (Hours):</label>
            <input type="number" id="causal_window" name="causal_window" step="0.5" value="{{ causal_window }}" required><br><br>
            
            <label for="min_threshold">Min Incidents (Threshold):</label>
            <input type="number" id="min_threshold" name="min_threshold" value="{{ min_threshold }}" required><br><br>

            <button type="submit">Run Analysis Pipeline</button>
        </form>
        <p style="margin-top: 20px; font-size: 12px;">Default High Dependency Weight: 0.9</p>
    </div>

    <div class="main-content">
        {% if error %}
            <p class="error">An Error Occurred: {{ error }}</p>
        {% elif result %}
            <h2>Pipeline Results</h2>
            {{ result | safe }}
            
            <h2>Risk Breakdown</h2>
            <table class="risk-table">
                <tr><th>Department</th><th>Risk Level</th><th>Probability</th><th>Revenue Exposure</th><th>Mitigation Steps</th></tr>
                {% for item in breakdown %}
                    <tr>
                        <td>{{ item.Department }}</td>
                        <td class="{{ item['Risk Level'].split(' ')[0] }}">{{ item['Risk Level'] }}</td>
                        <td>{{ "%.4f"|format(item['Risk Probability']) }}</td>
                        <td>{{ item['Unresolved Revenue Exposure (₹)'] }}</td>
                        <td>
                            <details>
                                <summary>View Steps</summary>
                                <p>{{ item['Mitigation Steps'] }}</p>
                            </details>
                        </td>
                    </tr>
                {% endfor %}
            </table>
            
            <h2>Dynamic Dependency Graph (Learned)</h2>
            <pre>{{ graph_json | safe }}</pre>
        {% else %}
            <p>Set parameters and click 'Run Analysis Pipeline' to start the prediction.</p>
        {% endif %}
    </div>
</div>
"""

@app.route("/", methods=["GET"])
def index():
    # Initial load of the page
    return render_template_string(HTML_TEMPLATE, causal_window=3.0, min_threshold=10)

@app.route("/run", methods=["POST"])
def run_pipeline_frontend():
    # Get parameters from the form
    causal_window = request.form.get("causal_window", type=float, default=3.0)
    min_threshold = request.form.get("min_threshold", type=int, default=10)

    # Prepare data for the FastAPI call
    payload = {
        "causal_window": causal_window,
        "min_threshold": min_threshold
    }

    try:
        # Call the FastAPI backend
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        # Get the structured result from the API
        api_result = response.json()

        if api_result.get('error'):
            return render_template_string(HTML_TEMPLATE, error=api_result['error'], causal_window=causal_window, min_threshold=min_threshold)

        # Convert the markdown summary to HTML for display
        html_summary = markdown.markdown(api_result['summary'])
        
        # Format the dependency graph nicely
        graph_json = json.dumps(api_result['dependency_graph'], indent=4)
        
        return render_template_string(
            HTML_TEMPLATE,
            result=html_summary,
            breakdown=api_result['department_risks'],
            graph_json=graph_json,
            causal_window=causal_window,
            min_threshold=min_threshold
        )

    except requests.exceptions.RequestException as e:
        return render_template_string(HTML_TEMPLATE, error=f"API Connection Error: Could not connect to FastAPI backend at {API_URL}. Is it running on port 8000?", causal_window=causal_window, min_threshold=min_threshold)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e), causal_window=causal_window, min_threshold=min_threshold)

if __name__ == "__main__":
    # Run Flask on the default port 5000
    # Use: python app.py
    print("Run using: python app.py (Flask frontend on port 5000)")
    app.run(debug=True, port=5000)