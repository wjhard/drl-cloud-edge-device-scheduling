from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_STATS_PATH = Path(__file__).with_name("normalization_stats.json")


class ObservationNormalizer:
    def __init__(self, stats_path: str | Path = DEFAULT_STATS_PATH):
        self.stats_path = Path(stats_path)
        self.stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        self.epsilon = float(self.stats.get("epsilon", 1e-8))

    def normalize(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized = {
            key: np.array(value, copy=True)
            for key, value in observation.items()
        }

        if "task_features" in normalized:
            self._normalize_task_features(normalized)
        if "resource_features" in normalized:
            self._normalize_feature_columns(
                normalized["resource_features"],
                self.stats.get("resource_features", {}),
            )
        if "global_features" in normalized:
            self._normalize_feature_columns(
                normalized["global_features"],
                self.stats.get("global_features", {}),
            )
        return normalized

    def _normalize_task_features(self, observation: dict[str, np.ndarray]) -> None:
        task_features = observation["task_features"]
        valid_mask = observation.get("task_valid_mask")
        if valid_mask is None:
            valid_rows = np.ones(task_features.shape[0], dtype=bool)
        else:
            valid_rows = valid_mask.astype(bool)

        for column_text, stat in self.stats.get("task_features", {}).items():
            column = int(column_text)
            mean = float(stat["mean"])
            std = float(stat["std"])
            task_features[valid_rows, column] = (
                task_features[valid_rows, column] - mean
            ) / (std + self.epsilon)

    def _normalize_feature_columns(self, features: np.ndarray, stats: dict[str, dict]) -> None:
        for column_text, stat in stats.items():
            column = int(column_text)
            mean = float(stat["mean"])
            std = float(stat["std"])
            features[..., column] = (features[..., column] - mean) / (std + self.epsilon)
