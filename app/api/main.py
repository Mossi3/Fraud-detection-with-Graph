import os
import json
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from app.train import train_model, ARTIFACTS_DIR
from app.models.gnn import HeteroSAGE


app = FastAPI(title="Graph Fraud Rings API")


class TrainRequest(BaseModel):
    use_mock: bool = True
    csv_path: Optional[str] = None
    num_transactions: int = 20000
    seed: int = 42


class Transaction(BaseModel):
    transaction_id: str
    card_id: str
    merchant_id: str
    device_id: str
    ip: str
    amount: float
    timestamp: str


class InferRequest(BaseModel):
    transactions: List[Transaction]


_CACHE: Dict[str, Any] = {
    "scores": None,
    "id_maps": None,
    "metrics": None,
}


def _load_artifacts_into_cache() -> None:
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    rings_path = os.path.join(ARTIFACTS_DIR, "rings.json")
    idmaps_path = os.path.join(ARTIFACTS_DIR, "id_maps.json")
    scores_path = os.path.join(ARTIFACTS_DIR, "card_scores.npy")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            _CACHE["metrics"] = json.load(f)
    if os.path.exists(idmaps_path):
        with open(idmaps_path, "r") as f:
            _CACHE["id_maps"] = json.load(f)
    if os.path.exists(scores_path):
        _CACHE["scores"] = np.load(scores_path)
    if os.path.exists(rings_path):
        with open(rings_path, "r") as f:
            _CACHE["rings"] = json.load(f)


@app.post("/train")
def train(req: TrainRequest):
    result = train_model(
        use_mock=req.use_mock,
        csv_path=req.csv_path or "",
        num_transactions=req.num_transactions,
        seed=req.seed,
    )
    _load_artifacts_into_cache()
    return result


@app.get("/metrics")
def metrics():
    _load_artifacts_into_cache()
    return _CACHE.get("metrics", {"message": "No metrics available. Train first."})


@app.get("/rings")
def rings():
    _load_artifacts_into_cache()
    return _CACHE.get("rings", [])


@app.post("/infer")
def infer(req: InferRequest):
    _load_artifacts_into_cache()
    if _CACHE.get("scores") is None or _CACHE.get("id_maps") is None:
        return {"error": "Model not trained. Call /train first."}
    scores = _CACHE["scores"]
    id_maps = _CACHE["id_maps"]
    card_map: Dict[str, int] = id_maps.get("card", {})
    result = []
    for tx in req.transactions:
        idx = card_map.get(tx.card_id)
        score = float(scores[idx]) if idx is not None else 0.5
        result.append({"transaction_id": tx.transaction_id, "card_id": tx.card_id, "fraud_score": round(score, 6)})
    return {"scores": result}

