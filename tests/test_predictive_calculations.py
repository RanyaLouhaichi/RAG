# tests/test_predictive_calculations.py
import pytest
from datetime import datetime, timedelta
from agents.predictive_analysis_agent import PredictiveAnalysisAgent

class TestPredictiveCalculations:
    """Test the actual calculation methods"""
    
    @pytest.fixture
    def agent(self):
        return PredictiveAnalysisAgent()
    
    @pytest.fixture
    def sample_tickets(self):
        """Create sample tickets with various states"""
        tickets = []
        base_date = datetime.now() - timedelta(days=10)
        
        # Completed tickets
        for i in range(10):
            tickets.append({
                "key": f"DONE-{i}",
                "fields": {
                    "status": {"name": "Done"},
                    "resolutiondate": (base_date + timedelta(days=i)).isoformat() + "Z",
                    "created": (base_date - timedelta(days=5)).isoformat() + "Z",
                    "assignee": {"displayName": f"Dev{i % 3}"}
                },
                "changelog": {
                    "histories": [{
                        "created": (base_date + timedelta(days=i-2)).isoformat() + "Z",
                        "items": [{"field": "status", "toString": "In Progress"}]
                    }, {
                        "created": (base_date + timedelta(days=i)).isoformat() + "Z",
                        "items": [{"field": "status", "toString": "Done"}]
                    }]
                }
            })
        
        # In progress tickets
        for i in range(5):
            tickets.append({
                "key": f"WIP-{i}",
                "fields": {
                    "status": {"name": "In Progress"},
                    "assignee": {"displayName": f"Dev{i % 3}"}
                }
            })
        
        # To do tickets
        for i in range(5):
            tickets.append({
                "key": f"TODO-{i}",
                "fields": {
                    "status": {"name": "To Do"},
                    "assignee": {"displayName": f"Dev{i % 3}"}
                }
            })
        
        return tickets
    
    def test_velocity_history_extraction(self, agent, sample_tickets):
        """Test that velocity history is correctly extracted"""
        history = agent._extract_velocity_history(sample_tickets)
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert all(isinstance(v, (int, float)) for v in history)
        
    def test_sprint_burndown_calculation(self, agent, sample_tickets):
        """Test burndown calculation"""
        burndown = agent._calculate_sprint_burndown_data(sample_tickets)
        
        assert 'ideal_burndown' in burndown
        assert 'actual_burndown' in burndown
        assert 'predicted_burndown' in burndown
        assert 'current_velocity' in burndown
        assert 'on_track' in burndown
        
        # Verify burndown arrays make sense
        assert burndown['ideal_burndown'][0] == burndown['total_tickets']
        assert burndown['ideal_burndown'][-1] == 0
        
    def test_capacity_forecast_calculation(self, agent, sample_tickets):
        """Test capacity forecast calculation"""
        metrics = {
            "workload": {
                "Dev0": 7,
                "Dev1": 6,
                "Dev2": 7
            }
        }
        
        capacity = agent._calculate_capacity_forecast(sample_tickets, metrics)
        
        assert 'team_size' in capacity
        assert 'total_capacity' in capacity
        assert 'member_capacities' in capacity
        assert 'capacity_forecast' in capacity
        
        # Verify member capacities
        assert len(capacity['member_capacities']) == capacity['team_size']
        
        # Verify forecast
        assert len(capacity['capacity_forecast']) == 4  # 4 weeks
        
    def test_sprint_completion_probability(self, agent, sample_tickets):
        """Test sprint completion probability calculation"""
        metrics = {
            "throughput": 10,
            "cycle_time": 3
        }
        historical_data = {
            "velocity_history": [8, 9, 10, 10, 11]
        }
        
        result = agent._calculate_sprint_completion_probability(
            sample_tickets, metrics, historical_data
        )
        
        assert 'probability' in result
        assert 'confidence' in result
        assert 'risk_level' in result
        assert 'remaining_work' in result
        assert 'recommended_actions' in result
        
        # Probability should be between 0 and 1
        assert 0 <= result['probability'] <= 1
        
        # Risk level should be valid
        assert result['risk_level'] in ['minimal', 'low', 'medium', 'high', 'critical']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])