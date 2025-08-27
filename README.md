# Flight Delay Prediction API

[![Deployment](https://img.shields.io/badge/Render-Deployment-blueviolet)](https://flight-un-known.onrender.com)

An end-to-end machine learning project that predicts US airline flight delays. This repository contains the code for a FastAPI application, containerized with Docker and deployed to the cloud.

---
### 🚀 Live API

The API is deployed on Render and is available at the following URL:

**[https://flight-un-known.onrender.com](https://flight-un-known.onrender.com)**

You can test the live API via the interactive documentation here:

**[https://flight-un-known.onrender.com/docs](https://flight-un-known.onrender.com/docs)**


---
## ## Project Overview

The goal of this project was to build a complete machine learning application from scratch. This involved sourcing and cleaning data, training a classification model, and deploying it as a public-facing API. The model predicts whether a given flight is likely to be delayed by 15 minutes or more.

Key challenges included handling a large, imbalanced dataset, managing Python dependencies, debugging a live application, and solving real-world deployment issues.

---
## ## Tech Stack

* **Backend:** Python, FastAPI
* **Machine Learning:** Pandas, Scikit-learn, LightGBM
* **Containerization:** Docker
* **Deployment:** Render, Git, GitHub

---
## ## Model Performance

The final LightGBM model was trained and evaluated with a focus on correctly identifying delayed flights (optimizing for **Recall**).

| Metric | Score (on Test Set) |
| :--- | :--- |
| **Accuracy** | ~82% |
| **Precision (for Delayed Flights)** | ~55% |
| **Recall (for Delayed Flights)** | **~63%** |

The model successfully identifies approximately 63% of all delayed flights, providing a useful tool for travelers despite the inherent difficulty of the prediction task.

---
## ## How to Run Locally

To run this project on your local machine, you'll need Git and Docker installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Unknown2151/flight-delay-prediction.git](https://github.com/Unknown2151/flight-delay-prediction.git)
    cd flight-delay-predictor
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t flight-predictor-api .
    ```

3.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 flight-predictor-api
    ```

4.  **Access the application:**
    Open your web browser and go to `http://localhost:8000`.

---
## ## API Usage

The primary endpoint is `/predict`. You can send a `POST` request with form data to get a prediction.

**Endpoint:** `POST /predict`

**Form Field:**
* `flight_number` (string, required): The flight number you want to predict (e.g., "AA234").

**Example using cURL:**
```bash
curl -X POST -F "flight_number=AA234" http://localhost:8000/predict