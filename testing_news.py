import requests

# ✅ Localhost endpoint
url = "http://localhost:5000/predict"

# ✅ Sample news to test
data = {
    "text": "Breaking: The government has confirmed that aliens have landed in New York."
}

# ✅ Send POST request
response = requests.post(url, json=data)

# ✅ Print result
print("Response JSON:")
print(response.json())
