# Delhi AQI Intelligence

A portfolio-grade air-quality intelligence project for Delhi: CPCB-style AQI estimation, multi-point live-data aggregation, data-quality monitoring, and an uncertainty-aware one-hour forecast.

Live app: https://delhiaqiprediction-oqtovurfmqkjew82fym3nk.streamlit.app/

## Advanced capabilities

| Capability | Implementation |
|---|---|
| Multi-point data | Six Delhi geographic points are fetched concurrently and aggregated using a robust median. |
| AQI methodology | Pollutant sub-indices and dominant pollutant follow CPCB breakpoints. CO is converted from OpenWeather µg/m³ to mg/m³. |
| Data quality | Freshness, missing-pollutant checks, dataset size and an explainable quality score are shown in the dashboard. |
| Temporal ML | Hourly-normalized inputs use 1/2/3/6/12/24-hour lags, rolling levels, pollutant trends and cyclical time features. Hyperparameters are selected with walk-forward validation. |
| Uncertainty | Quantile gradient-boosting models plus split-conformal calibration provide a data-driven 80% forecast interval. |
| Honest baseline | The point model forecasts the one-hour AQI change and walk-forward validation tunes how much of that change to trust. Held-out MAE is compared with persistence, and the dashboard plots the latest 72 out-of-sample errors. |
| Reproducibility | Python and core ML dependencies are pinned; model metadata includes versions and temporal metrics. |
| MLOps | GitHub Actions collects data, retrains the model and commits the resulting artifact. |

## Architecture

Six Delhi grid points → median aggregation → versioned hourly dataset → CPCB AQI calculator + time-series point and quantile models → Streamlit dashboard.

## Forecast design

Inputs include current pollutant readings, cyclic hour/day-of-week variables, AQI lags through 24 hours, rolling AQI levels and short-term pollutant trends. Raw observations are normalized to robust hourly medians without filling missing hours. The newest 20% of history remains untouched for final evaluation; the system reports MAE, RMSE, persistence-baseline improvement and empirical interval coverage.

## Run locally

1. Clone this repository and create a Python 3.11 virtual environment.
2. Install dependencies: pip install -r requirements.txt
3. Set OPENWEATHER_API_KEY.
4. Run python fetch_api.py.
5. Run python train_forecaster.py.
6. Run streamlit run app.py.

## Important limitation

The app is an indicative CPCB-style estimate, not an official CPCB AQI feed. OpenWeather locations are geographic grid points rather than validated CPCB stations, and official AQI requires pollutant-specific averaging windows. For official-grade reporting, replace the collector with CPCB or CAQM station data and retain raw station-level observations.

## Project layout

- app.py — Streamlit intelligence dashboard
- aqi_calculator.py — CPCB-style sub-index calculator
- data_quality.py — freshness and completeness checks
- fetch_api.py — concurrent multi-point collector
- train_forecaster.py — time-series point and quantile forecast training
- models/ — generated model artifact and metadata
- .github/workflows/ — automated collection and training
