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
    is_fraud: bool

@app.post("/ingest")
def ingest_event(req: IngestRequest):
    global inference_enabled

    vec = np.array(req.features, dtype=float)
    buffer.add(vec, req.src_ip)  # buffer stores all traffic for cold-start count only

    # Only legitimate traffic goes to adaptive baseline buffer
    baseline.buffer_for_update(vec, req.src_ip, is_fraud=req.is_fraud)

    # Cold-start training only once when enough events have arrived
    if not inference_enabled and buffer.is_ready(min_samples=INIT_SAMPLES):
        X, ip_series = buffer.get_window_data()
        print(f"[TRAIN] Training on initial clean traffic (sample count={len(X)})")

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

    # Adaptive baseline update uses only non-fraud legitimate traffic
    baseline.update()
    X, ip_series = buffer.get_window_data()
    baseline.add_new_ips(X, ip_series)

    return {"status": "updated", "message": "Baseline adapted using clean traffic"}
