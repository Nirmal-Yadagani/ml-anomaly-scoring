import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class TrafficBuffer:
    def __init__(self, window: str = "1min"):
        """
        Buffers streaming ML feature vectors + IP labels until initial training window is satisfied.
        """
        self.window = window
        self.data: List[np.ndarray] = []
        self.ips: List[str] = []
        self.start_time = pd.Timestamp.utcnow()

    def add(self, feature_vec: np.ndarray, ip: str):
        """
        Add one extracted ML feature vector and its source IP.
        """
        self.data.append(np.array(feature_vec, dtype=float))
        self.ips.append(str(ip))

    def is_ready(self, min_samples: int = 200) -> bool:
        """
        Check if buffer has enough samples to begin training.
        """
        return len(self.data) >= min_samples

    def get_window_data(self) -> Tuple[np.ndarray, pd.Series]:
        """
        Returns:
          - Stacked numeric feature matrix (N x D)
          - Parallel IP labels (Series of length N)
        """
        if not self.data:
            return np.empty((0, 0)), pd.Series([], dtype=str)

        X = np.vstack(self.data)
        ip_series = pd.Series(self.ips)
        return X, ip_series

    def summary(self) -> Dict[str, int]:
        """
        Quick summary for the receiving team / logs.
        """
        return {"total_samples": len(self.data), "unique_ips": len(set(self.ips))}
