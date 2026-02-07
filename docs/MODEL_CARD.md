# Model Card: Parking Availability Prediction

## Overview

- **Purpose**: Predict zone-level parking occupancy rate (and derived free spaces) for a given time, with 90% prediction intervals.
- **Model type**: LightGBM regression (point estimate) plus quantile regression (0.1, 0.9) for intervals.
- **Outputs**: Occupancy rate [0, 1], free spaces, confidence interval, plain-language summary.

## Training data

- **Source**: Synthetic occupancy and inventory for prototype; replace with city open data (e.g. LADOT) for production.
- **Temporal range**: Configurable (e.g. 60–90 days). Time-based train/validation split (e.g. last 4 weeks held out).
- **Aggregation**: Zone-level occupancy per time bucket (e.g. 30 minutes).

## Features

- Temporal: hour, day of week, weekend flag, month.
- Lags: same-zone occupancy at t−1, t−2, t−3, t−4.
- Same hour previous week (seasonality).
- Traffic proxy: peak morning/evening flags (or real traffic when available).
- Events proxy: placeholder (0) until event calendar is connected.

## Evaluation

- **Metrics**: MAE, RMSE on held-out period. Baseline: same-hour-previous-week prediction.
- **Typical result**: ~25–30% MAE improvement over baseline on synthetic data; production metrics depend on data quality and city.

## Limitations

- Predictions are probabilistic and not a guarantee of availability.
- Accuracy depends on recency and quality of occupancy (and optional traffic/events) data.
- Best used for planning; users should re-check closer to arrival when possible.

## Intended use

- Commuter apps and dashboards for “where to park” guidance.
- City planners for capacity and policy analysis (with appropriate caveats).

## Out-of-scope

- Not for real-time enforcement or billing. Not validated for safety-critical decisions.
