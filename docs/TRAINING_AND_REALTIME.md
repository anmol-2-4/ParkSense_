# How We Train the Data and Predict in Real Time

## Training (offline, batch)

- **Input data**: Historical parking occupancy (per space, per time bucket) and zone inventory. For the prototype we use synthetic data; for production you plug in open data (e.g. [LADOT](https://catalog.data.gov/dataset/ladot-parking-meter-occupancy) historical archive).
- **Pipeline**:
  1. **Ingest** → raw occupancy + inventory.
  2. **Zones** → aggregate occupancy to zone-level per time bucket (e.g. every 30 min).
  3. **Features** → for each (zone, timestamp): time of day, day of week, weekend, month; **lags** (occupancy at t−1, t−2, t−3, t−4); same-hour-previous-week; traffic/events proxies.
  4. **Train/val split** → by time (e.g. last 4 weeks = validation).
  5. **Model** → LightGBM regression (point prediction) + quantile regression (0.1, 0.9) for 90% intervals. Model is saved to disk.
- **When**: Run on a schedule (e.g. daily) so the model sees recent patterns. No real-time data is used during training; it’s all batch historical.

So: **training = historical data → features → train model → save**. The AI (LightGBM) learns from past patterns.

---

## Prediction (online, can use real-time data)

- When a user asks “what’s availability now?” the **same saved model** is used, but the **inputs** (features) can come from:
  - **Current time** (hour, day, weekend, month, peak flags) — always.
  - **Latest occupancy** (the last 1–4 buckets per zone) — **if we have real-time or latest-observed data**.
  - **Same hour last week** — from a live feed or from the last stored snapshot.
- If we **do** feed the model with the latest observed occupancy (and optionally same-hour-last-week), the **AI prediction** is then driven by real-time data: the model was trained on historical data, but at inference it uses live inputs to produce the prediction.
- If we **don’t** have latest occupancy (e.g. no feed yet), we use **defaults** (e.g. 0.5 for lags) so the prediction is based only on time and calendar; it’s still an AI prediction, but not conditioned on current occupancy.

So: **real-time data = use live (or latest-observed) occupancy as features → run the trained AI model → show prediction (and optionally “current” occupancy) in the UI.**

---

## End-to-end flow

```
[Historical occupancy + inventory]
        ↓
  Zone aggregation
        ↓
  Feature building (temporal + lags + same_hour_prev_week + proxies)
        ↓
  Train LightGBM + quantile models  →  save model + “latest” snapshot
        ↓
[At request time]
  Current time + (optional) latest occupancy per zone from cache/feed
        ↓
  Same feature building (inference path)
        ↓
  Load model → predict → return occupancy, free spaces, interval, summary
        ↓
  UI shows “AI predicted” (and optionally “current observed”)
```

---

## What “real-time” means in this codebase

1. **Latest snapshot at training time**  
   After training we save the **most recent** zone-level occupancy (last 4 buckets per zone, and same-hour-previous-week). The API loads this and uses it as **last_known_occupancy** (and same_hour_prev_week) when predicting for “now”. So even without a live API, the model gets “most recent historical” as a proxy for current state.

2. **Optional live feed (production)**  
   You can replace or augment that snapshot with a **live occupancy feed** (e.g. LADOT real-time): a small job periodically fetches current occupancy, aggregates by zone, and updates a cache or file. The API then uses this cache when building features for “now”, so predictions are driven by **real-time data** and the **AI** (trained model) together.

3. **LADOT live mode (demo + production)**  
   When `realtime.mode=ladot_live`, the API pulls LADOT’s live feed on a timer and updates zone-level lags in memory. This provides real-time inputs without any external sensors beyond the open data feed.

So: we **train on historical data**; we **predict using the AI model**; and we **show/use real-time data** by feeding the latest occupancy (and optionally same-hour-last-week) into that model at request time.
