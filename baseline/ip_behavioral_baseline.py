import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from typing import Dict

def safe_inverse(cov: np.ndarray) -> np.ndarray:
    """Return stable inverse covariance, fallback if singular or invalid."""
    dim = cov.shape[0]
    if not np.all(np.isfinite(cov)):
        return np.eye(dim) * 1e3
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_reg = cov + np.eye(dim) * 1e-3
        try:
            return np.linalg.inv(cov_reg)
        except:
            return np.eye(dim) * 1e3

class IPBehavioralBaseline:
    def __init__(self, alpha: float = 0.1, decay_days: int = 7):
        self.ip_baseline: Dict[str, Dict] = {}
        self.fitted_ips: set[str] = set()
        self.alpha = alpha
        self.decay_days = decay_days
        self.clean_buffer: Dict[str, list[np.ndarray]] = {}  # stores only non-fraud traffic
        self.decay_rate = 0.15  # for confidence decay, does NOT corrupt stored stats

    # --------------------------------------------------
    # Cold start training per IP
    # --------------------------------------------------
    def fit(self, features: np.ndarray, ips: pd.Series):
        for ip in ips.unique():
            X = features[ips == ip]
            if len(X) < 1:
                continue

            dim = X.shape[1]
            mu = X.mean(axis=0)

            # Covariance for coupling
            cov = np.eye(dim) * 1e-3 if len(X) == 1 else np.cov(X.T) + np.eye(dim) * 1e-6
            cov_inv = safe_inverse(cov)

            # Compute MD distances and percentile threshold
            dists = [mahalanobis(x, mu, cov_inv) for x in X]
            md_p99 = float(np.percentile(dists, 99)) if dists else 1.0

            self.ip_baseline[ip] = {
                "mu": mu,
                "cov": cov,
                "cov_inv": cov_inv,
                "md_p99": md_p99,
                "last_retrain": pd.Timestamp.utcnow(),
                "n_samples": len(X),
            }

            self.fitted_ips.add(ip)

    # --------------------------------------------------
    # Inference score for 1 request
    # --------------------------------------------------
    def score(self, x: np.ndarray, ip: str) -> float:
        if ip not in self.fitted_ips:
            return 0.0
        b = self.ip_baseline[ip]
        md = mahalanobis(x, b["mu"], b["cov_inv"])
        return min(md / (b["md_p99"] + 1e-6), 1.0)

    # --------------------------------------------------
    # Buffer only clean traffic for adaptive update
    # Fraud samples are ignored here
    # --------------------------------------------------
    def buffer_for_update(self, x: np.ndarray, ip: str, is_fraud: bool = False):
        if is_fraud:  # ❌ never store fraud cases
            return
        self.clean_buffer.setdefault(ip, []).append(np.array(x, dtype=float))

    # --------------------------------------------------
    # Adaptive update per IP using only clean samples
    # --------------------------------------------------
    def update(self):
        """Adaptive update using only legitimate traffic drift per IP."""
        if not self.clean_buffer:
            return

        now = pd.Timestamp.utcnow()

        for ip, vecs in self.clean_buffer.items():
            if ip not in self.fitted_ips:
                continue

            X = np.vstack(vecs)
            if X.size == 0:
                continue

            b = self.ip_baseline[ip]
            dim = X.shape[1]

            # New coupled stats
            new_mu = X.mean(axis=0)
            new_cov = np.cov(X.T) + np.eye(dim) * 1e-6 if len(X) > 1 else np.eye(dim) * 1e-3

            # EMA blend μ and Σ
            b["mu"] = (1 - self.alpha) * b["mu"] + self.alpha * new_mu
            b["cov"] = (1 - self.alpha) * b["cov"] + self.alpha * new_cov

            # Recompute inverse covariance
            b["cov_inv"] = safe_inverse(b["cov"])

            # Recompute Mahalanobis threshold
            dists = [mahalanobis(x, b["mu"], b["cov_inv"]) for x in X]
            if dists:
                b["md_p99"] = float(np.percentile(dists, 99))

            # Update lifecycle fields
            b["last_retrain"] = now
            b["n_samples"] = len(X)

            self.ip_baseline[ip] = b

        # Clear clean drift buffer after update
        self.clean_buffer.clear()

    # --------------------------------------------------
    # Add new IPs if they appear later
    # --------------------------------------------------
    def add_new_ips(self, features: np.ndarray, ips: pd.Series):
        new_ips = set(ips.unique()) - self.fitted_ips
        for ip in new_ips:
            X = features[ips == ip]
            if len(X) > 0:
                self.fit(X, ips)

    # --------------------------------------------------
    # Confidence decay influences scoring only
    # Does NOT modify core baseline distribution
    # --------------------------------------------------
    def decay_confidence(self):
        now = pd.Timestamp.utcnow()
        for ip in self.fitted_ips:
            b = self.ip_baseline[ip]
            age = (now - b["last_retrain"]).days
            if age > self.decay_days:
                factor = math.exp(-self.decay_rate * (age - self.decay_days))
                b["md_p99"] *= factor  # decayed influence during scoring
