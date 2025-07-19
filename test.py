# test_forecast_manual.py
import requests
import json
from pprint import pprint

# Base URL of your API
BASE_URL = "http://localhost:5001"

def test_velocity_forecast():
    """Test velocity forecast endpoint"""
    print("\n=== Testing Velocity Forecast ===")
    response = requests.post(
        f"{BASE_URL}/api/forecast/JURIX",  # Use a real project
        json={"type": "velocity"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Type: {data['type']}")
        
        if 'data' in data:
            forecast_data = data['data']
            print(f"\nTrend: {forecast_data.get('trend')}")
            print(f"Confidence: {forecast_data.get('confidence')}")
            print(f"Next Week Estimate: {forecast_data.get('next_week_estimate')}")
            print(f"Current Velocity: {forecast_data.get('current_velocity')}")
            
            if 'chart' in forecast_data:
                chart = forecast_data['chart']
                print(f"\nChart Labels: {chart['data']['labels']}")
                print(f"Data Points: {chart['data']['datasets'][0]['data']}")
        else:
            print("No forecast data returned")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_burndown_forecast():
    """Test burndown forecast endpoint"""
    print("\n=== Testing Burndown Forecast ===")
    response = requests.post(
        f"{BASE_URL}/api/forecast/MG",
        json={"type": "burndown"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        
        if 'data' in data:
            burndown_data = data['data']
            print(f"\nCompletion Probability: {burndown_data.get('completion_probability')}")
            print(f"Days Remaining: {burndown_data.get('days_remaining')}")
            print(f"At Risk: {burndown_data.get('at_risk')}")
            print(f"On Track: {burndown_data.get('on_track')}")
            
            print(f"\nIdeal Burndown: {burndown_data.get('ideal_burndown')}")
            print(f"Predicted Burndown: {burndown_data.get('predicted_burndown')}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_capacity_forecast():
    """Test capacity forecast endpoint"""
    print("\n=== Testing Capacity Forecast ===")
    response = requests.post(
        f"{BASE_URL}/api/forecast/MG",
        json={"type": "capacity"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        
        if 'data' in data:
            capacity_data = data['data']
            print(f"\nCurrent Capacity: {capacity_data.get('current_capacity')}")
            print(f"Optimal Capacity: {capacity_data.get('optimal_capacity')}")
            print(f"Utilization: {capacity_data.get('utilization_percentage')}%")
            print(f"Capacity Trend: {capacity_data.get('capacity_trend')}")
            print(f"At Risk Members: {capacity_data.get('at_risk_members')}")
            
            if 'overloaded_members' in capacity_data:
                print("\nOverloaded Members:")
                for member in capacity_data['overloaded_members']:
                    print(f"  - {member['name']}: {member['current_load']} tickets")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test all three forecast types
    test_velocity_forecast()
    test_burndown_forecast()
    test_capacity_forecast()