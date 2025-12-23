from locust import HttpUser, task, between

class FlightApiUser(HttpUser):
    # Simulate a user waiting between 1 and 5 seconds between requests
    wait_time = between(5, 10)

    @task
    def test_health(self):
        self.client.get("/health")

    @task(3) # This task is 3x more likely to run than the others
    def test_prediction(self):
        # We send a dummy request to see how the server handles logic
        payload = {
            "carrierCode": "AA",
            "flightNumber": "234",
            "scheduledDepartureDate": "2025-10-23"
        }
        self.client.post("/predict", json=payload)