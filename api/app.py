from fastapi import FastAPI
import numpy as np
from typing import List
import pandas as pd
from pydantic import BaseModel

from ingestion.traffic_buffer import TrafficBuffer
from baseline.ip_behavioral_baseline import IPBehavioralBaseline
from models.isolation_forest import IsolationForestModel
from scoring.hybrid_scorer import HybridScorer

app = FastAPI(title="ML Anomaly Scoring Service")

# --------------------------------------------------
# Initialize models + buffer
# --------------------------------------------------

INIT_SAMPLES = 200  # cold start window
UPDATE_EVERY = 200  # adaptive update frequency

buffer = TrafficBuffer(window="1min")
baseline = IPBehavioralBaseline(alpha=0.1)
iforest = IsolationForestModel(contamination=0.02)
scorer = HybridScorer(if_weight=0.6, baseline_weight=0.4, threshold=0.3)

inference_enabled = False


# --------------------------------------------------
# API Endpoints
# --------------------------------------------------


class IngestRequest(BaseModel):
    features: List[float]
    src_ip: str

@app.post("/ingest")
def ingest_event(req: IngestRequest):
    global inference_enabled
    vec = np.array(req.features, dtype=float)
    buffer.add(vec, req.src_ip)

    if not inference_enabled and buffer.is_ready(min_samples=INIT_SAMPLES):
        X, ip_series = buffer.get_window_data()
        baseline.fit(X, ip_series)
        iforest.fit(X)
        inference_enabled = True
        return {"status": "trained", "message": "Cold-start training complete"}

    return {"status": "buffering", "samples": buffer.summary()}


class ScoreRequest(BaseModel):
    features: List[float]
    src_ip: str

@app.post("/score")
def score_event(req: ScoreRequest):
    if not inference_enabled:
        return {"anomaly_score": 0.0, "status": "not_ready"}
    vec = np.array(req.features, dtype=float)
    return {"anomaly_score": scorer.score(vec, req.src_ip, iforest, baseline), "status": "inference"}



@app.post("/update")
def adaptive_update():
    global inference_enabled
    if not inference_enabled:
        return {"status": "not_ready", "message": "Models not trained yet"}

    X, ip_series = buffer.get_window_data()
    baseline.update(X, ip_series)
    baseline.add_new_ips(X, ip_series)
    return {"status": "updated", "message": "Baseline adapted"}


# --------------------------------------------------
# Local streaming simulation for your testing
# --------------------------------------------------

if __name__ == "__main__":
    import random
    import time

    print("🚀 Running local streaming test with buffer integration...\n")

    test_ips = ["192.168.1.10", "10.0.0.5", "127.0.0.1"]

    for i in range(1000):
        vec = np.array([
            random.uniform(5, 120),      # req_rate
            random.uniform(1, 20),      # unique_uri_count
            random.uniform(20, 200),    # payload_size_mean
            random.uniform(0.1, 4.0),   # payload_entropy
            random.uniform(0, 0.3),     # error_rate_4xx
            random.uniform(0, 0.2),     # error_rate_5xx
            random.uniform(0, 1.0),     # burstiness
            random.uniform(0.001, 0.1), # endpoint_rarity
            random.uniform(0.1, 2.0),   # interarrival_mean
            random.uniform(0.1, 2.5),   # interarrival_std
            random.uniform(10, 500),    # avg_response_time
        ])

        ip = random.choice(test_ips)
        buffer.add(vec, ip)

        if not inference_enabled and buffer.is_ready(min_samples=INIT_SAMPLES):
            X, ip_series = buffer.get_window_data()
            print(f"[TRAIN] Training triggered at t={i} (samples={len(X)})")
            baseline.fit(X, ip_series)
            iforest.fit(X)
            inference_enabled = True
            print("[TRAIN] Initial training complete. Now scoring...\n")

        if inference_enabled:
            score = scorer.score(vec, ip, iforest, baseline)
            if score > scorer.threshold:
                print(f"[ALERT] t={i}  IP={ip}  anomaly_score={score}")

        if i > 0 and i % UPDATE_EVERY == 0 and buffer.is_ready(min_samples=INIT_SAMPLES):
            X, ip_series = buffer.get_window_data()
            print(f"[UPDATE] Adaptive update at t={i}")
            baseline.update(X, ip_series)
            baseline.add_new_ips(X, ip_series)

        # time.sleep(0.01)

    print("\n✅ Local streaming test finished successfully.")
