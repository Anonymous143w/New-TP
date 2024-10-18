import requests
import json

def test_api():
    # Test data
    test_data = {
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 2,
        "smoker": "no",
        "region": "southwest"
    }
    
    # Make prediction
    response = requests.post(
        "http://localhost:8000/predict",
        json=test_data
    )
    
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api()