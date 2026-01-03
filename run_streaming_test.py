import time
import random
import numpy as np
import pandas as pd
from baseline.ip_behavioral_baseline import IPBehavioralBaseline
from models.isolation_forest import IsolationForestModel
from scoring.hybrid_scorer import HybridScorer
from feature_engineering.extractor import FeatureExtractor
from ingestion.schema import TrafficEvent
from ingestion.traffic_buffer import TrafficBuffer

# --------------------------------------------------
# Initialize components
# --------------------------------------------------

INIT_SAMPLES = 200  # cold-start training window size
UPDATE_EVERY = 200  # adaptive baseline update frequency

buffer = TrafficBuffer(window="1min")
baseline = IPBehavioralBaseline(alpha=0.1)
iforest = IsolationForestModel(contamination=0.02)
scorer = HybridScorer(if_weight=0.6, baseline_weight=0.4, threshold=0.3)
extractor = FeatureExtractor(window="1min")

print("🚀 Streaming simulation started...\n")

# --------------------------------------------------
# Generate synthetic streaming events
# --------------------------------------------------

ips = ["192.168.1.10", "10.0.0.5", "127.0.0.1"]
ua_pool = ["bot-agent", "mobile-agent", "browser-agent"]

# We'll stream 10,000 events one by one
events = (
    TrafficEvent(
        timestamp=pd.Timestamp.utcnow(),
        src_ip=random.choice(ips),
        uri_path="/api/login",
        method="POST",
        payload_size=random.uniform(20, 200),
        response_time_ms=random.uniform(10, 500),
        status_code=random.choice([200, 404, 429, 500]),
        user_agent=random.choice(ua_pool),
    )
    for _ in range(10_000)
)

# --------------------------------------------------
# Streaming loop: buffer → train → infer → update
# --------------------------------------------------

for i, event in enumerate(events):
    # Extract ML features for this one event (single row)
    feats = extractor.extract([event])["ml_features"]
    vec = feats.values[0]  # numeric feature vector
    ip = event.src_ip

    # Add to traffic buffer
    buffer.add(vec, ip)

    # -------------------------
    # Cold-start training
    # -------------------------
    if not baseline.fitted_ips and buffer.is_ready(min_samples=INIT_SAMPLES):
        X, ip_series = buffer.get_window_data()
        print(f"[TRAIN] Collected {len(X)} samples. Training models now...")
        baseline.fit(X, ip_series)
        iforest.fit(X)
        print("[TRAIN] Cold-start training complete. Inference mode enabled.\n")

    # -------------------------
    # Inference scoring
    # -------------------------
    if ip in baseline.fitted_ips:
        score = scorer.score(vec, ip, iforest, baseline)
        if score > scorer.threshold:
            print(f"[ALERT] t={i}  IP={ip}  anomaly_score={score}")

    # -------------------------
    # Adaptive baseline update
    # -------------------------
    if i > 0 and i % UPDATE_EVERY == 0 and buffer.is_ready(min_samples=INIT_SAMPLES):
        X, ip_series = buffer.get_window_data()
        print(f"[UPDATE] Adaptive baseline update at t={i} (window size={len(X)})")
        baseline.update(X, ip_series)
        baseline.add_new_ips(X, ip_series)

    time.sleep(0.001)  # simulate high-speed but not instant

print("\n✅ Streaming simulation finished.")
