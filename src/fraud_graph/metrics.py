from __future__ import annotations

from typing import Dict, List

import numpy as np


def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, np.ndarray]:
    y_true = y_true.astype(int)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    total_pos = max(1, int(y_true.sum()))

    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / total_pos

    thresholds = y_score_sorted
    # prepend point (0,1) style not required for AP computation here
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if int(y_true.sum()) == 0:
        return 0.0
    pr = _precision_recall_curve(y_true, y_score)
    precision = pr["precision"]
    recall = pr["recall"]
    # Average precision: sum over recall steps of precision
    # Compute recall deltas
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    delta = recall - recall_prev
    ap = float(np.sum(precision * delta))
    return ap


def precision_recall_points(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, List[float]]:
    pr = _precision_recall_curve(y_true, y_score)
    return {
        "precision": pr["precision"].tolist(),
        "recall": pr["recall"].tolist(),
        "thresholds": pr["thresholds"].tolist(),
    }

