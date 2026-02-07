# Parking Availability Prediction

**Zone-wise parking availability predictions** using time, location, traffic proxies, and event signals. Built for smart-city deployment and commuter applications—open data only, with confidence intervals and a production-ready API.

---

## Features

- **Predictions by zone**: Occupancy rate, free spaces, and 90% confidence intervals
- **Plain-language summaries**: e.g. “High confidence—good time to plan to park here”
- **Open data**: Prototype uses synthetic data; designed to plug in [LADOT](https://catalog.data.gov/dataset/ladot-parking-meter-occupancy) or other city feeds
- **REST API**: Typed request/response models, OpenAPI docs at `/docs` and `/redoc`
- **Commuter UI**: Map and zone list with one-click focus

---

## Quick start

```bash
./run_demo.sh
```

Then open **http://127.0.0.1:8000** for the UI, or **http://127.0.0.1:8000/docs** for the API documentation.

**Real-time demo (Los Angeles):** By default the API runs a **LADOT live feed** that pulls real-time meter occupancy and refreshes every 10 seconds. See `config/default.yaml` → `realtime` to switch to `simulated` or `snapshot` mode.

---

## Manual setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.train.run
.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## API overview

| Endpoint | Description |
|----------|-------------|
| `GET /` | Commuter web UI (map + zone list) |
| `GET /health` | Health check and model status |
| `GET /zones` | Zone metadata (centroid, capacity) |
| `GET /predict?zone_id={id}` | Single-zone prediction with confidence |
| `GET /predict/all` | Predictions for all zones |
| `GET /metrics` | Training metrics and baseline comparison |
| `GET /docs` | OpenAPI (Swagger) documentation |
| `GET /redoc` | ReDoc documentation |

All prediction responses include `occupancy_rate`, `free_spaces`, `confidence_interval_90`, and a human-readable `summary`.

---

## Training vs real-time prediction

- **Training** uses **historical** occupancy (and inventory) in batch: aggregate by zone, build features (time, lags, same-hour-last-week, proxies), then train and save the model. Run on a schedule (e.g. daily) so the model stays up to date.
- **Prediction** uses the **saved model** plus **latest observed** occupancy when available: at request time we feed the model with current time and the most recent zone occupancy (from a snapshot written at training time, or from a live feed). So you get **real-time data** (latest occupancy) as input and **AI prediction** (LightGBM) as output. The UI shows both “predicted” and “observed” when the latest snapshot is present.

See [Training and real-time](docs/TRAINING_AND_REALTIME.md) for the full flow.

## Model card

A short [model card](docs/MODEL_CARD.md) describes the model type, training data, features, evaluation, and intended use.

---

## Project structure

```
├── config/
│   └── default.yaml          # City, time bucket, features, model hyperparameters
├── src/
│   ├── ingest/               # Data ingestion (synthetic; add LADOT or other sources)
│   ├── zones/                # Zone definition and occupancy aggregation
│   ├── features/             # Temporal, lags, traffic/events proxies
│   ├── train/                # LightGBM + quantile training pipeline
│   ├── model/                # Load model and run inference with confidence
│   └── api/                  # FastAPI app, Pydantic schemas
├── static/                   # Commuter UI (map + sidebar)
├── docs/
│   └── MODEL_CARD.md         # Model card (metrics, limitations, intended use)
├── data/                     # Created at runtime (raw, processed, models)
├── run_demo.sh
├── requirements.txt
└── README.md
```

---

## Data sources (production)

- **Parking occupancy**: [LADOT Parking Meter Occupancy](https://catalog.data.gov/dataset/ladot-parking-meter-occupancy) (real-time), [LADOT Archive](https://catalog.data.gov/dataset/ladot-parking-meter-occupancy-archive) (historical)
- **Zone geography**: [LADOT Metered Parking Inventory](https://catalog.data.gov/dataset/ladot-metered-parking-inventory-policies)
- **Traffic / events**: Configure city-specific open APIs or proxies in `config/default.yaml`

---

## Adding another city

1. Provide occupancy data (e.g. `space_id`, `timestamp`, `occupied`) and inventory (`space_id`, `lat`, `lon`, and a zone or cluster id).
2. Point config data paths to your raw and processed directories.
3. Run `python -m src.train.run`. For production, run training on a schedule (e.g. daily).

---

## License

MIT. See [LICENSE](LICENSE). Intended for civic and smart-city use.
