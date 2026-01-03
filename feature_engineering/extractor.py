import pandas as pd
import numpy as np
from typing import List, Dict
from scipy.stats import entropy

from ingestion.schema import TrafficEvent


class FeatureExtractor:
    def __init__(self, window: str = "1min"):
        self.window = window

    def events_to_df(self, events: List[TrafficEvent]) -> pd.DataFrame:
        """Convert TrafficEvent list to DataFrame"""
        df = pd.DataFrame([e.__dict__ for e in events])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        return df

    def extract(self, events: List[TrafficEvent]) -> Dict[str, pd.DataFrame]:
        """
        Main entry point.
        Returns:
          {
            context,
            behavioral_features,
            ml_features
          }
        """
        df = self.events_to_df(events)
        df = self._add_temporal_features(df)

        behavioral = self._compute_behavioral_features(df)
        ml_features = self._select_ml_features(behavioral)

        return {
            "context": df.set_index("timestamp"),
            "behavioral_features": behavioral,
            "ml_features": ml_features
        }


    # ------------------------------------------------------------------
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["time_of_day"] = df["timestamp"].dt.hour
        df["interarrival"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
        return df

    def _compute_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling-window behavioral features.
        Assumes df has timestamp index.
        """

        df = df.set_index("timestamp")

        # Encode categorical columns for rolling ops
        df["_uri_code"], _ = pd.factorize(df["uri_path"])
        df["_method_code"], _ = pd.factorize(df["method"])

        # -------------------------------
        # Numeric-only dataframe
        # -------------------------------
        numeric_cols = [
            "payload_size",
            "status_code",
            "response_time_ms",
            "interarrival",
        ]

        df_num = df[numeric_cols]

        rolling = df_num.rolling(self.window)

        features = pd.DataFrame(index=df.index)

        # -------------------------------
        # Volume / rate
        # -------------------------------
        features["req_count"] = df["uri_path"].rolling(self.window).count()
        features["req_rate"] = (
            features["req_count"]
            / pd.to_timedelta(self.window).total_seconds()
            * 60
        )


        features["unique_uri_count"] = (
            df["_uri_code"]
            .rolling(self.window)
            .apply(lambda x: len(set(x)), raw=True)
        )

        features["unique_method_count"] = (
            df["_method_code"]
            .rolling(self.window)
            .apply(lambda x: len(set(x)), raw=True)
        )

        # -------------------------------
        # Payload statistics
        # -------------------------------
        features["payload_size_mean"] = rolling["payload_size"].mean()
        features["payload_size_std"] = rolling["payload_size"].std()
        features["payload_size_max"] = rolling["payload_size"].max()

        # -------------------------------
        # Payload entropy
        # -------------------------------
        features["payload_entropy"] = (
            df["payload_size"]
            .rolling(self.window)
            .apply(
                lambda x: entropy(np.bincount(x.astype(int)))
                if len(x) > 1
                else 0,
                raw=False,
            )
        )

        # -------------------------------
        # Error rates
        # -------------------------------
        features["error_rate_4xx"] = (
            df["status_code"]
            .rolling(self.window)
            .apply(lambda x: ((x >= 400) & (x < 500)).mean(), raw=False)
        )

        features["error_rate_5xx"] = (
            df["status_code"]
            .rolling(self.window)
            .apply(lambda x: (x >= 500).mean(), raw=False)
        )

        # -------------------------------
        # Response time
        # -------------------------------
        features["avg_response_time"] = rolling["response_time_ms"].mean()

        # -------------------------------
        # Temporal behavior
        # -------------------------------
        features["interarrival_mean"] = rolling["interarrival"].mean()
        features["interarrival_std"] = rolling["interarrival"].std()

        features["burstiness"] = (
            features["interarrival_std"]
            / features["interarrival_mean"].replace(0, np.nan)
        ).fillna(0)

        # -------------------------------
        # Endpoint rarity (global)
        # -------------------------------
        endpoint_freq = df["uri_path"].value_counts(normalize=True)
        features["endpoint_rarity"] = df["uri_path"].map(
            lambda x: 1 / endpoint_freq.get(x, 1)
        )

        # -------------------------------
        # Cleanup
        # -------------------------------
        features = features.fillna(0)
        df.drop(columns=["_uri_code", "_method_code"], inplace=True)

        return features

    def _select_ml_features(self, behavioral: pd.DataFrame) -> pd.DataFrame:
        """Final ML feature vector"""
        return behavioral[
            [
                "req_rate",
                "unique_uri_count",
                "payload_size_mean",
                "payload_entropy",
                "error_rate_4xx",
                "error_rate_5xx",
                "avg_response_time",
                "endpoint_rarity",
                "interarrival_mean",
                "interarrival_std",
                "burstiness",
            ]
        ]
