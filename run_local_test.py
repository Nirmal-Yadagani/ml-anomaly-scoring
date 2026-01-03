import time, random
import numpy as np
import pandas as pd

from baseline.ip_behavioral_baseline import IPBehavioralBaseline
from models.isolation_forest import IsolationForestModel
from scoring.hybrid_scorer import HybridScorer
from feature_engineering.extractor import FeatureExtractor
from ingestion.schema import TrafficEvent

# Simulated team streaming input contract
extractor = FeatureExtractor(window="1min")

# Generate synthetic TrafficEvent objects
ips = ["192.168.1.10", "10.0.0.5", "127.0.0.1"]
ua_pool = ["bot-agent", "mobile-agent", "browser-agent"]
events = [
    TrafficEvent(
        timestamp=pd.Timestamp.utcnow(),
        src_ip=random.choice(ips),
        uri_path="/api/login",
        method="POST",
        payload_size=random.uniform(20, 200),
        response_time_ms=random.uniform(10, 500),
        status_code=random.choice([200, 404, 429, 500]),
        user_agent=random.choice(ua_pool)
    )
    for _ in range(10000)
]

output = extractor.extract(events)
ml_features = output["ml_features"]
ip_series = output["context"]["src_ip"]

# Cold start training
bl = IPBehavioralBaseline()
ifm = IsolationForestModel()
scorer = HybridScorer()

X = ml_features.values
bl.fit(X, ip_series)
ifm.fit(X)

# Streaming inference + adaptive updates
for i in range(len(ml_features)):
    vec = ml_features.iloc[i].values
    ip = str(ip_series.iloc[i])  # <-- always scalar, no hash/unhashable issue, no non-unique index issue

    score = scorer.score(vec, ip, ifm, bl)
    if score > 0.3:
        print(f"[ALERT] {ip} anomaly score={score}")

    if i % 200 == 0:
        bl.update(X, ip_series)

    time.sleep(0.001)

print("done")
