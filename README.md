# 🌫️ Delhi AQI Predictor

<div align="center">

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://delhiaqiprediction-oqtovurfmqkjew82fym3nk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A production-grade machine learning application that predicts Delhi's Air Quality Index (AQI) in real-time using live pollutant and weather data.**

[🚀 Live Demo](https://delhiaqiprediction-oqtovurfmqkjew82fym3nk.streamlit.app/) · [📊 View Notebook](notebooks/) · [🐛 Report Bug](https://github.com/harshsharma5468/Delhi_AQI_prediction/issues)

</div>

---

## 🎯 Problem Statement

Delhi consistently ranks among the world's most polluted cities. AQI levels fluctuate dramatically by hour, season, and location — yet most people have no accessible, real-time way to understand what they're breathing or plan their day accordingly.

This project builds a live ML-powered tool to:
- Predict current and near-future AQI from real pollutant readings
- Classify air quality into health risk categories
- Visualize historical pollution trends and patterns
- Provide actionable health recommendations

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔴 **Real-Time Prediction** | Fetches live pollutant data via API and predicts AQI instantly |
| 📈 **Historical Analysis** | Visualize PM2.5, PM10, NO2, SO2, CO, O3 trends over time |
| 💊 **Health Recommendations** | Personalized advice based on predicted AQI category |
| 🔁 **Automated Data Pipeline** | GitHub Actions fetches fresh data every hour automatically |
| 🐳 **Dockerized Deployment** | Fully containerized — runs identically anywhere |
| 📊 **Model Insights** | Feature importance charts and model performance metrics |

---

## 🏗️ Project Architecture

```
Delhi_AQI_prediction/
│
├── .github/workflows/       # CI/CD — automated hourly data fetch + tests
├── data/                    # Raw and processed AQI datasets
├── models/                  # Trained ML model files (.pkl)
├── notebooks/               # EDA, feature engineering, model training
├── src/                     # Core ML pipeline (training, prediction)
├── utils/                   # Helper functions (API calls, preprocessing)
│
├── app.py                   # Main Streamlit application
├── main.py                  # Entry point for pipeline execution
├── fetch_api.py             # Real-time API data fetching
├── live_data_collector.py   # Automated live data collection
├── hourly_fetch.yml         # GitHub Actions schedule config
│
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service container orchestration
├── Makefile                 # Build, run, and test commands
├── setup.sh                 # Environment setup script
├── deploy.sh                # One-command deployment script
└── requirements.txt         # Python dependencies
```

---

## 🤖 Machine Learning

### Model Performance

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Random Forest ⭐ | **0.94** | **8.2** | **12.4** |
| XGBoost | 0.92 | 9.1 | 13.8 |
| Linear Regression | 0.78 | 18.6 | 24.3 |

> Random Forest selected as final model based on best generalization performance.

### Features Used

| Feature | Description |
|---|---|
| `PM2.5` | Fine particulate matter (µg/m³) |
| `PM10` | Coarse particulate matter (µg/m³) |
| `NO2` | Nitrogen dioxide (µg/m³) |
| `SO2` | Sulphur dioxide (µg/m³) |
| `CO` | Carbon monoxide (mg/m³) |
| `O3` | Ozone (µg/m³) |
| `hour` | Hour of day (cyclical encoding) |
| `month` | Month (cyclical encoding) |
| `day_of_week` | Day of week (cyclical encoding) |

### AQI Categories

| AQI Range | Category | Health Impact |
|---|---|---|
| 0 – 50 | 🟢 Good | Minimal impact |
| 51 – 100 | 🟡 Satisfactory | Minor breathing discomfort |
| 101 – 200 | 🟠 Moderate | Discomfort to sensitive people |
| 201 – 300 | 🔴 Poor | Discomfort to most people |
| 301 – 400 | 🟣 Very Poor | Respiratory illness likely |
| 401+ | ⚫ Severe | Affects healthy people too |

---

## ⚙️ Automated Data Pipeline

```
Every Hour (GitHub Actions)
        │
        ▼
fetch_api.py → Live Pollutant API
        │
        ▼
live_data_collector.py → Preprocess & Validate
        │
        ▼
aqi_data.csv → Append new records
        │
        ▼
Model Prediction → Streamlit Dashboard
```

The pipeline runs on a schedule defined in `.github/workflows/hourly_fetch.yml`, ensuring the dashboard always shows current data without manual intervention.

---

## 🚀 Getting Started

### Option 1: Streamlit Cloud (No Setup Required)
👉 **[Click here for the live demo](https://delhiaqiprediction-oqtovurfmqkjew82fym3nk.streamlit.app/)**

### Option 2: Run with Docker

```bash
# Clone the repository
git clone https://github.com/harshsharma5468/Delhi_AQI_prediction.git
cd Delhi_AQI_prediction

# Build and run with Docker Compose
docker-compose up --build
```

App will be available at `http://localhost:8501`

### Option 3: Run Locally

```bash
# Clone and setup
git clone https://github.com/harshsharma5468/Delhi_AQI_prediction.git
cd Delhi_AQI_prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 4: Using Makefile

```bash
make setup     # Install dependencies
make run       # Start Streamlit app
make docker    # Build and run Docker container
make test      # Run tests
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
API_KEY=your_air_quality_api_key_here
```

> Get a free API key from [OpenWeatherMap](https://openweathermap.org/api) or [WAQI](https://aqicn.org/api/).

---

## 📊 Key Results & Insights

- **Model achieves 94% R² score** on test data using Random Forest
- **PM2.5 and PM10** are the strongest predictors of AQI (>60% feature importance)
- **Winter months (Nov–Jan)** show 3x higher average AQI than summer months
- **Peak pollution hours** are 7–10 AM and 8–11 PM (traffic correlation)
- **Automated pipeline** collects and stores data every hour with zero manual effort

---

## 🛠️ Tech Stack

**Machine Learning:** Python, Scikit-learn, XGBoost, Pandas, NumPy

**Visualization:** Streamlit, Matplotlib, Seaborn, Plotly

**MLOps:** Docker, Docker Compose, GitHub Actions (CI/CD)

**Data:** Live Air Quality API, Historical CPCB Delhi data

**Dev Tools:** Makefile, Shell Scripts, Git

---

## 📁 Notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_EDA.ipynb` | Exploratory data analysis and visualizations |
| `notebooks/02_feature_engineering.ipynb` | Feature creation and selection |
| `notebooks/03_model_training.ipynb` | Model comparison and hyperparameter tuning |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

**Harsh Sharma**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/harsh-sharma)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/harshsharma5468)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail)](mailto:sharsh2808@gmail.com)

---

<div align="center">
  <b>If this project helped you, please give it a ⭐ on GitHub!</b>
</div>
