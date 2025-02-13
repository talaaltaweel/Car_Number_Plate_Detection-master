
import datetime

# Function to detect anomalies in vehicle behavior
def detect_anomaly(vehicle):
    if vehicle["location"] == "Restricted Area" or vehicle["speed"] > 80:
        return True
    return False

# Sample vehicle data
vehicle_data = {
    "vehicle_id": "123ABC",
    "location": "Restricted Area",
    "speed": 100,
    "timestamp": datetime.datetime.now()
}

# Check for anomalies
if __name__ == "__main__":
    if detect_anomaly(vehicle_data):
        print(f"Anomaly Detected for Vehicle {vehicle_data['vehicle_id']}")

def detect_anomaly(vehicle_data):
    # Example logic for detecting anomalies
    return True  # Modify this based on your conditions

vehicle_data = {"vehicle_id": "123ABC", "speed": 80, "location": "Jordan"}

if __name__ == "__main__":
    if detect_anomaly(vehicle_data):
        print(f"Anomaly Detected for Vehicle {vehicle_data['vehicle_id']}")