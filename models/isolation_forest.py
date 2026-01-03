from sklearn.ensemble import IsolationForest

class IsolationForestModel:
    def __init__(self, contamination=0.02, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.fitted = False

    def fit(self, X):
        self.model.fit(X)
        self.fitted = True

    def score(self, x):
        if not self.fitted:
            return 0.0
        return -self.model.decision_function([x])[0]
