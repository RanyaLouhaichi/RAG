from typing import Dict, Any
from flask import Flask, request, jsonify
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.core.productivity_workflow import run_productivity_workflow # type: ignore
import logging
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/dashboard', methods=['POST'])
def run_dashboard():
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid or missing JSON datain request body"}), 400

        project_id = data.get("project_id")
        if not project_id:
            return jsonify({"error": "Missing required field: project_id"}), 400

        time_range = data.get("time_range", {})
        if not isinstance(time_range, dict) or not time_range.get("start") or not time_range.get("end"):
            return jsonify({"error": "Invalid or missing time_range fields (start and end required)"}), 400

        logging.info(f"Received input data: {data}")
        logging.info(f"Running dashboard workflow for project_id={project_id}, time_range={time_range}")
        state = run_productivity_workflow(project_id, time_range)

        response = {
            "project_id": state["project_id"],
            "time_range": state["time_range"],
            "tickets": state["tickets"],
            "metrics": state["metrics"],
            "visualization_data": state["visualization_data"],
            "recommendations": state["recommendations"],
            "report": state["report"],
            "workflow_status": state["workflow_status"],
            "dashboard_id": state.get("dashboard_id")
        }

        if state["workflow_status"] == "success":
            return jsonify(response), 200
        else:
            return jsonify({"error": "Workflow failed", **response}), 500

    except Exception as e:
        logging.error(f"Error in dashboard endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)