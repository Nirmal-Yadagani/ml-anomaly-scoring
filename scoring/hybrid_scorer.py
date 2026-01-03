import numpy as np

class HybridScorer:
    def __init__(self, if_weight=0.6, baseline_weight=0.4, threshold=0.75):
        self.if_weight = if_weight
        self.baseline_weight = baseline_weight
        self.threshold = threshold

    def score(self, x: np.ndarray, ip: str, if_model, bl_model) -> float:
        if_score = if_model.score(x)
        bl_score = bl_model.score(x, ip)
        return round(self.if_weight * if_score + self.baseline_weight * bl_score, 3)
