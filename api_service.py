import os
import json
import random
import requests

def get_flight_price(origin: str, destination: str, trip_type: str, currency: str) -> str:
    api_endpoint = "https://www.mocky.io/v2/5c3c7d2d300000980a055e76"  # Thay bằng URL Mocky.io thực tế
    trip_name = "khứ hồi" if trip_type == "round_trip" else "một chiều"
    
    try:
        response = requests.get(api_endpoint, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        flights = data.get("flights", [])
        # Tìm chuyến bay khớp với origin, destination, và trip_type
        for flight in flights:
            if (flight.get("origin") == origin and 
                flight.get("destination") == destination and 
                flight.get("trip_type") == trip_type):
                price = flight.get("price", random.randint(1500000, 5000000))
                return f"Giá vé máy bay {trip_name} từ {origin} đến {destination} khoảng {price:,} {currency}."
        
        # Nếu không tìm thấy chuyến bay khớp
        return f"Không tìm thấy chuyến bay {trip_name} từ {origin} đến {destination}."
    
    except requests.exceptions.RequestException as e:
        try:
            # Fallback: Read from local mock_data.json
            with open(os.path.join("data", "mock_data.json"), "r", encoding="utf-8") as file:
                data = json.load(file)
            flights = data.get("flights", [])
            for flight in flights:
                if (flight.get("origin") == origin and 
                    flight.get("destination") == destination and 
                    flight.get("trip_type") == trip_type):
                    price = flight.get("price", random.randint(1500000, 5000000))
                    return f"Giá vé máy bay {trip_name} từ {origin} đến {destination} khoảng {price} {currency}."
            price = random.randint(1500000, 5000000)
            return f"Không tìm thấy chuyến bay {trip_name} từ {origin} đến {destination}."
        
        except (FileNotFoundError, json.JSONDecodeError) as file_error:
            return f"Giá vé máy bay {trip_name} từ {origin} đến {destination} khoảng {random.randint(1500000, 5000000):,} {currency} (dữ liệu dự phòng do lỗi khi đọc file mock_data.json: {str(file_error)})."