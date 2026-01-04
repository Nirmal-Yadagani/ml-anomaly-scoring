import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from typing import Dict, List

from ingestion.db_reader import PostgresReader
from feature_engineering.extractor import FeatureExtractor
from ingestion.schema import TrafficEvent

def safe_inverse(cov: np.ndarray) -> np.ndarray:
    dim = cov.shape[0]
    if not np.all(np.isfinite(cov)):
        return np.eye(dim) * 1e3
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_reg = cov + np.eye(dim)*1e-3
        try:
            return np.linalg.inv(cov_reg)
        except:
            return np.eye(dim) * 1e3

class IPBehavioralBaseline:
    def __init__(self, dsn: str, alpha=0.1):
        self.dsn = dsn
        self.ip_baseline: Dict[str, Dict] = {}
        self.fitted_ips: set = set()
        self.alpha = alpha
        self.decay_rate = 0.05
        self.extractor = FeatureExtractor(window="1min")
        self.is_ready = False

    def fit(self, ip: str, n: int = 2000):
        reader = PostgresReader(self.dsn)
        df = reader.fetch(f"""
            SELECT * FROM traffic_2026_01_02
            WHERE src_ip = '{ip}' AND model_label = 0
            ORDER BY timestamp DESC
            LIMIT {n};
        """)
        if df.empty:
            return False
        
        # DB rows → TrafficEvent objects
        events: List[TrafficEvent] = [
            TrafficEvent(
                timestamp=row.timestamp,
                src_ip=row.src_ip,
                method=row.method,
                uri_path=row.uri_path,
                status_code=int(row.status_code),
                payload_size=int(row.payload_size),
                response_time_ms=float(row.response_time_ms),
                user_agent=row.user_agent
            )
            for row in df.itertuples()
        ]

        extracted = self.extractor.extract(events)
        ml: pd.DataFrame = extracted["ml_features"]

        X = ml.values
        mu = X.mean(axis=0)
        cov = np.cov(X.T) + np.eye(X.shape[1])*1e-6
        cov_inv = safe_inverse(cov)

        dists = [mahalanobis(x, mu, cov_inv) for x in X]
        md_p99 = float(np.percentile(dists, 99)) if dists else 1.0

        self.ip_baseline[ip] = {"mu":mu,"cov":cov,"cov_inv":cov_inv,"md_p99":md_p99,"last_retrain":pd.Timestamp.utcnow(),"n_samples":len(df)}
        self.fitted_ips.add(ip)
        self.is_ready = True
        return True

    def score(self, ip: str, n):
        if not self.is_ready:
            return 0.0
        reader = PostgresReader(self.dsn)
        df = reader.fetch(f"""
            SELECT * FROM traffic_2026_01_02
            WHERE src_ip = '{ip}' AND model_label = 0
            ORDER BY timestamp DESC
            LIMIT {n};
        """)
        if df.empty:
            return False
        
        # DB rows → TrafficEvent objects
        events: List[TrafficEvent] = [
            TrafficEvent(
                timestamp=row.timestamp,
                src_ip=row.src_ip,
                method=row.method,
                uri_path=row.uri_path,
                status_code=int(row.status_code),
                payload_size=int(row.payload_size),
                response_time_ms=float(row.response_time_ms),
                user_agent=row.user_agent
            )
            for row in df.itertuples()
        ]

        extracted = self.extractor.extract(events)
        ml: pd.DataFrame = extracted["ml_features"]
        X = ml.values
        if ip not in self.fitted_ips:
            self.fit(ip)
            return 0.0
        b = self.ip_baseline[ip]
        md = mahalanobis(X, b["mu"], b["cov_inv"])
        return min(md/(b["md_p99"]+1e-6),1.0)

    def update(self, ip: str, n: int = 2000):
        """EMA update using fresh DB window for this IP"""
        if ip not in self.fitted_ips:
            return self.fit(ip, n)

        reader = PostgresReader(self.dsn)
        df = reader.fetch(f"""
            SELECT * FROM traffic_2026_01_02
            WHERE src_ip = '{ip}' AND model_label = 0
            ORDER BY timestamp DESC
            LIMIT {n};
        """)
        if df.empty:
            return False

        # DB rows → TrafficEvent objects
        events: List[TrafficEvent] = [
            TrafficEvent(
                timestamp=row.timestamp,
                src_ip=row.src_ip,
                method=row.method,
                uri_path=row.uri_path,
                status_code=int(row.status_code),
                payload_size=int(row.payload_size),
                response_time_ms=float(row.response_time_ms),
                user_agent=row.user_agent
            )
            for row in df.itertuples()
        ]

        extracted = self.extractor.extract(events)
        ml: pd.DataFrame = extracted["ml_features"]

        X = ml.values
        b = self.ip_baseline[ip]

        # EMA update μ and Σ
        b["mu"] = (1-self.alpha)*b["mu"] + self.alpha*X.mean(axis=0)
        new_cov = np.cov(X.T) + np.eye(X.shape[1])*1e-6
        b["cov"] = (1-self.alpha)*b["cov"] + self.alpha*new_cov
        b["cov_inv"] = safe_inverse(b["cov"])

        # Update md_p99
        dists = [mahalanobis(x, b["mu"], b["cov_inv"]) for x in X]
        if dists:
            b["md_p99"] = float(np.percentile(dists,99))

        # Confidence decay
        age = (pd.Timestamp.utcnow() - b["last_retrain"]).days
        if age > 7:
            b["md_p99"] *= math.exp(-self.decay_rate*(age-7))

        b["last_retrain"] = pd.Timestamp.utcnow()
        b["n_samples"] = len(df)
        self.ip_baseline[ip] = b
        return True
