import requests
BASE_URL = "http://localhost:8000"
def test_get_products():
    response = requests.get(f"{BASE_URL}/products")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
def test_predict_sales():
    payload = {
    "product_id": 1,
    "unit_price": 10.5,
    "month": 4,
    "year": 2025,
    "TotalRevenue": 1500.0
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_quantity" in json_data
    assert "input_details" in json_data

