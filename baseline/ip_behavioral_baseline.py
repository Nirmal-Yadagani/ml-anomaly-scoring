import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from typing import Dict, List

def safe_inverse(cov):
    dim = cov.shape[0]
    if not np.all(np.isfinite(cov)):
        return np.eye(dim) * 1e3
    try:
        return np.linalg.inv(cov)
    except:
        cov_reg = cov + np.eye(dim)*1e-3
        try:
            return np.linalg.inv(cov_reg)
        except:
            return np.eye(dim) * 1e3

class IPBehavioralBaseline:
    def __init__(self, alpha=0.1):
        self.ip_basline: Dict[str, Dict] = {}
        self.fitted_ips = set()
        self.alpha = alpha

    def fit(self, features: np.ndarray, ips: pd.Series):
        for ip in ips.unique():
            group = features[ips == ip]
            dim = features.shape[1]
            if len(group) <= 1:
                cov = np.eye(dim) * 1e-3
            else:
                cov = np.cov(group.T) + np.eye(dim)*1e-6
            mu = group.mean(axis=0)
            cov_inv = safe_inverse(cov)
            dists = [mahalanobis(x, mu, cov_inv) for x in group]
            md_p99 = float(np.percentile(dists, 99)) if dists else 1.0
            self.ip_basline[ip] = {"mu":mu,"cov":cov,"cov_inv":cov_inv,"md_p99":md_p99,"last_retrain":pd.Timestamp.utcnow(),"n_samples":len(group)}
            self.fitted_ips.add(ip)

    def score(self, x: np.ndarray, ip: str) -> float:
        if ip not in self.fitted_ips:
            return 0.0
        b = self.ip_basline[ip]
        md = mahalanobis(x, b["mu"], b["cov_inv"])
        return min(md/(b["md_p99"]+1e-6),1.0)

    def update(self, features: np.ndarray, ips: pd.Series):
        now = pd.Timestamp.utcnow()
        for ip in ips.unique():
            if ip not in self.fitted_ips:
                continue
            group = features[ips == ip]
            if group.size == 0:
                continue
            b = self.ip_basline[ip]
            new_mu = group.mean(axis=0)
            dim = new_mu.shape[0]
            new_cov = np.cov(group.T)+np.eye(dim)*1e-6 if len(group)>1 else np.eye(dim)*1e-3
            b["mu"]=(1-self.alpha)*b["mu"]+self.alpha*new_mu
            b["cov"]=(1-self.alpha)*b["cov"]+self.alpha*new_cov
            b["cov_inv"]=safe_inverse(b["cov"])
            dists=[mahalanobis(x,b["mu"],b["cov_inv"]) for x in group]
            if dists: b["md_p99"]=float(np.percentile(dists,99))
            b["last_retrain"]=now
            b["n_samples"]=len(group)
            self.ip_basline[ip]=b

    def add_new_ips(self, features: np.ndarray, ips: pd.Series):
        new_ips=set(ips.unique())-self.fitted_ips
        if new_ips:
            self.fit(features, ips)
