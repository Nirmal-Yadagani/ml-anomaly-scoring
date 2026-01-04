from fastapi import FastAPI
import numpy as np
from typing import List
from pydantic import BaseModel
import pandas as pd
import asyncio

from ingestion.traffic_buffer import TrafficBuffer
from baseline.ip_behavioral_baseline import IPBehavioralBaseline
from models.isolation_forest import IsolationForestModel
from scoring.hybrid_scorer import HybridScorer
from ingestion.db_reader import PostgresReader

app = FastAPI(title="ML Anomaly Scoring Service")

# Postgres connection
LOCAL_DSN = 'postgresql://postgres:postgres@localhost:5432/traffic_db'

db = PostgresReader(LOCAL_DSN)

# Runtime models
baseline_model = IPBehavioralBaseline(LOCAL_DSN, alpha=0.1)
iforest_model = IsolationForestModel(LOCAL_DSN, contamination=0.02)
# hybrid_model = HybridScorer(if_weight=0.6, baseline_weight=0.4, threshold=0.3)

buffer = TrafficBuffer(window="1min")
inference_enabled = False
ips: List[str] = []

@app.on_event("startup")
def cold_start():
    global inference_enabled, ips
    print("[BOOT] Loading IP list from DB")
    df_ips = db.fetch("""
        SELECT src_ip FROM traffic_2026_01_02
        GROUP BY src_ip
        ORDER BY MAX(timestamp) DESC
        LIMIT 50;
    """)
    ips = df_ips["src_ip"].tolist() if not df_ips.empty else []

    print("[BOOT] Training global Isolation Forest from DB")
    iforest_model.fit(10000)

    print("[BOOT] Training per-IP baseline from DB")
    for ip in ips:
        baseline_model.fit(ip, 2000)

    inference_enabled = True
    print("[BOOT] Models ready, inference enabled")

class IngestRequest(BaseModel):
    features: List[float]
    src_ip: str
    is_fraud: bool


@app.post("/ingest")
def ingest_event(req: IngestRequest):
    buffer.add(np.array(req.features, dtype=float), req.src_ip)
    return {"status": "ok", "buffer": buffer.summary()}

class ScoreRequest(BaseModel):
    n_if: int
    n_bsl: int
    src_ip: str

@app.post("/score")
def score_event(req: ScoreRequest):
    if not inference_enabled:
        return {"score": 0.0, "status": "not_ready"}
    s1 = iforest_model.score(req.n_if)
    s2 = baseline_model.score(req.src_ip, req.n_bsl)
    return {"score": float(0.6*abs(s1)-5/95 + 0.4*s2), "status": "inference"}

@app.post("/update_baseline_ip")
def update_baseline_ip(ip: str, samples: int):
    ok = baseline_model.update(ip, samples)
    return {"status": "updated"} if ok else {"status": "skipped"}


class RetrainConfig(BaseModel):
    train_limit: int = 10000
    min_clean: int = 200  # iForest clean window

@app.post("/retrain_iforest")
def retrain_iforest(cfg: RetrainConfig):
    """Retrain Isolation Forest using legitimate samples from DB"""
    reader = PostgresReader(LOCAL_DSN)
    df = reader.fetch(f"""
        SELECT * FROM traffic_2026_01_02
        WHERE model_label = 0
        ORDER BY timestamp DESC
        LIMIT {cfg.train_limit};
    """)

    if df.empty:
        return {"status": "error", "message": "No data in DB"}

    X = df.select_dtypes(include=[np.number]).values

    if len(X) < cfg.min_clean:
        return {"status": "skipped", "message": "Not enough clean data for iForest"}

    iforest_model.model.fit(X[: cfg.min_clean])
    iforest_model.ready = True

    return {"status": "retrained", "samples_used": int(len(X))}


@app.on_event("startup")
async def adaptive_loop():
    global inference_enabled
    if not inference_enabled:
        print("[BOOT] Inference not enabled yet — adaptive loop skipped")
        return

    async def updater():
        print("[BOOT] Adaptive per-IP baseline updater started")
        while True:
            for ip in baseline_model.fitted_ips:
                baseline_model.update(ip, 2000)
                print(f"[BASELINE UPDATE] Updated baseline for IP={ip}")
            print("[SLEEP] Next update in 30 seconds...\n")
            await asyncio.sleep(30)

    asyncio.create_task(updater())

