# ✈️ Flight Delay Predictor API v2.0

[![Deployment](https://img.shields.io/badge/Render-Live-blueviolet)](https://flight-un-known.onrender.com)
[![Python Tests](https://github.com/Unknown2151/flight-delay-prediction/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Unknown2151/flight-delay-prediction/actions)
[![API Version](https://img.shields.io/badge/API-v2.0-brightgreen)]()

An enterprise-grade **FastAPI** service that predicts US flight delays using **LightGBM**. Engineered for high availability with **Redis caching**, **Circuit Breaker** resilience, and automated **CI/CD**.

---

## 🚀 Key Engineering Highlights

### ⚡ High-Performance Caching

Uses **Render Redis (Key-Value)** to store prediction results.

- **Latency Reduction**: Cuts response times from **~3s** (external API fetch) to **<100ms** for cached flights.
- **Smart TTL**: 30-minute expiration ensures data freshness while reducing external API costs by ~80%.

### 🛡️ Fault-Tolerant Architecture

Built to survive external service outages (Amadeus/Tomorrow.io):

- **Circuit Breaker Pattern**: Automatically "trips" to prevent system hanging when external APIs fail.
- **Graceful Degradation**: Returns predictions based on default coordinates (JFK-LAX) if live data is unavailable, maintaining a **0% crash rate**.

### 📈 Proven Scalability

Validated via **Locust** load testing under 100+ concurrent users:

- **Success Rate**: **96.11%** (Exceeding the 80% industry-standard target).
- **Throughput**: ~125 requests/second handled via Gunicorn multi-worker architecture.

---

## 🏗️ System Architecture

```text
Request → [FastAPI Layer] → [Redis Cache Check] → HIT? → Response (<100ms)
               ↓ MISS
        [Circuit Breaker]
               ↓
    ┌──────────┴──────────┐
 [Amadeus API]     [Tomorrow.io API] → [ML Pipeline (LightGBM)] → [Cache & Respond]
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Backend** | Python 3.11, FastAPI, Pydantic v2, Gunicorn/Uvicorn |
| **Infrastructure** | Render (PaaS), Render Key-Value (Redis 7), Docker |
| **Machine Learning** | LightGBM (Optimized for 63% Recall), Scikit-learn, Pandas |
| **DevOps** | GitHub Actions, Pytest, Locust, Flake8 |

---

## 📡 API Usage: POST /predict

Validates airline codes (IATA) and prevents past-date queries using strict Pydantic schemas.

**Request Body:**

```json
{
  "carrierCode": "AA",
  "flightNumber": "100",
  "scheduledDepartureDate": "2026-06-01"
}
```

**Successful Response:**

```json
{
  "predicted_delay_status": 1,
  "predicted_delay_probability": "67.89%",
  "is_cached": false,
  "live_weather_at_origin": { "temperature": 18.5, "windSpeed": 12.0 },
  "timestamp": "2026-03-30T18:00:00"
}
```

---

## 🚀 Local Setup

**Clone & Install:**

```bash
git clone https://github.com/Unknown2151/flight-delay-prediction.git
cd flight-delay-prediction
pip install -r requirements.txt
```

**Environment:** Create a `.env` file with your `AMADEUS_API_KEY` and `TOMORROW_API_KEY`.

**Redis:** Ensure a local Redis instance is running:

```bash
docker run -p 6379:6379 -d redis
```

**Launch:**

```bash
uvicorn main:app --reload
```

**Docs:** Access the interactive Swagger UI at http://localhost:8000/docs.

---

## ✅ CI/CD Pipeline

The repository uses GitHub Actions to automate the production lifecycle:

- **Linting**: Enforces PEP8 standards via Flake8.
- **Unit Tests**: Executes Pytest suite with 80%+ coverage.
- **Integration Tests**: Spins up a Redis container to verify caching logic.
- **Auto-Deploy**: Triggers a production build on Render upon merging to main.

---

## 🤝 Support & License

Built with ❤️ by Sushmin. Licensed under MIT.

Found an issue? [Open a ticket here](https://github.com/Unknown2151/flight-delay-prediction/issues).

---

## 📚 Next Steps

Would you like me to help you generate a **"Project Case Study"** for your portfolio website? This would go deeper into the "Why" behind the Circuit Breaker and Redis choices to show off your architectural thinking.


