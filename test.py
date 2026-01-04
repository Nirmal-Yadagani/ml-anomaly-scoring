import requests

# The URL where your app is running (usually localhost:8000)
url = "http://127.0.0.1:8000/score"

# Data matching your ScoreRequest BaseModel
payload = {
    "n_if": 10,
    "n_bsl": 5,
    "src_ip": "192.168.1.1"
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Check for HTTP errors
    
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")