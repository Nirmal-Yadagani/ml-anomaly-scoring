from ingestion.traffic_buffer import TrafficBuffer
from baseline.ip_behavioral_baseline import IPBehavioralBaseline
from models.isolation_forest import IsolationForestModel
from scoring.hybrid_scorer import HybridScorer
import pandas as pd
import numpy as np

class AdaptiveTrainer:
    def __init__(self, init_samples=50):
        self.buffer = TrafficBuffer()
        self.baseline = IPBehavioralBaseline()
        self.if_model = IsolationForestModel()
        self.scorer = HybridScorer()
        self.init_samples = init_samples
        self.ready = False

    def add_event(self, vec, ip):
        self.buffer.add(vec, ip)
        X, ips = self.buffer.get_window_data()
        if len(X) >= self.init_samples and not self.ready:
            print("[INFO] Initial training started")
            self.baseline.fit(X, ips)
            self.if_model.fit(X)
            self.ready = True
            self.ready = True
            print("[INFO] Initial training completed")

    def infer(self, vec, ip):
        if not self.ready:
            return 0.0
        return self.scorer.score(vec, ip, self.if_model, self.baseline)

    def adaptive_update(self):
        X, ips = self.buffer.get_window_data()
        self.baseline.update(X, ips)
        self.baseline.add_new_ips(X, ips)
        print("[INFO] Baseline updated adaptively")
