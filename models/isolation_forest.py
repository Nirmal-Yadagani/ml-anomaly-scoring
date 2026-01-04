from sklearn.ensemble import IsolationForest
import numpy as np
from ingestion.db_reader import PostgresReader

class IsolationForestModel:
    def __init__(self, dsn: str, contamination=0.02):
        self.dsn = dsn
        self.model = IsolationForest(contamination=contamination)
        self.ready = False

    def fit(self, n: int = 10_000):
        reader = PostgresReader(self.dsn)
        df = reader.fetch(f"""
            SELECT * FROM requests
            WHERE is_fraud = false
            ORDER BY timestamp DESC
            LIMIT {n};
        """)
        if df.empty:
            return False
        self.model.fit(df.select_dtypes(include=[np.number]).values)
        self.ready = True
        return True

    def score(self, x: np.ndarray) -> float:
        return float(self.model.score_samples([x])[0])
