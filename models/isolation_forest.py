from sklearn.ensemble import IsolationForest
import numpy as np
from ingestion.db_reader import PostgresReader
from feature_engineering.extractor import FeatureExtractor
from ingestion.schema import TrafficEvent
from typing import List
import pandas as pd

class IsolationForestModel:
    def __init__(self, dsn: str, contamination=0.02):
        self.dsn = dsn
        self.model = IsolationForest(contamination=contamination)
        self.ready = False
        self.extractor = FeatureExtractor(window="1min")  # reuse extractor

    def fit(self, n: int = 10000):
        reader = PostgresReader(self.dsn)
        df = reader.fetch(f"""
            SELECT * FROM traffic_2026_01_02
            WHERE model_label = 0
            ORDER BY timestamp DESC
            LIMIT {n};
        """)

        if df.empty:
            return False
        
        # Convert DB rows → TrafficEvent objects
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

        # Extract ML feature matrix
        extracted = self.extractor.extract(events)
        ml_features: pd.DataFrame = extracted["ml_features"]

        self.model.fit(ml_features.values)
        self.ready = True
        return True

    def score(self, n) -> float:
        if not self.ready:
            return 0.0
        reader = PostgresReader(self.dsn)
        df = reader.fetch(f"""
            SELECT * FROM traffic_2026_01_02
            ORDER BY timestamp DESC
            LIMIT {n};
        """)

        if df.empty:
            return False
        
        # Convert DB rows → TrafficEvent objects
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

        # Extract ML feature matrix
        extracted = self.extractor.extract(events)
        ml_features: pd.DataFrame = extracted["ml_features"]
        if not self.ready:
            return 0.0
        return float(self.model.score_samples(ml_features.values)[0])
