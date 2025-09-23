from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import numpy as np
import torch

from src.fraud_graph.data import load_transactions
from src.fraud_graph.graph import build_indexer_from_df, build_adjacency, transactions_to_pairs
from src.fraud_graph.model import GraphSAGEModel, predict_scores
from src.fraud_graph.train import run_training
from src.fraud_graph.viz import fraud_heatmap, export_cluster_graph_json
from src.fraud_graph.community import build_weighted_graph, louvain_clusters, compute_cluster_stats


app = FastAPI(title="Graph Fraud Detection")


ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

STATE: Dict = {
    "model": None,
    "adj": None,
    "deg": None,
    "by_type": None,
    "offsets": None,
    "threshold": 0.55,
    "clusters": [],
    "edges_w": [],
}


class TrainRequest(BaseModel):
    data_path: str = "data/mock_transactions.csv"
    epochs: int = 8
    threshold: float = 0.55


class PredictRequest(BaseModel):
    transaction_id: str
    card_id: str
    merchant_id: str
    device_id: str
    ip_address: str
    amount: float
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train(req: TrainRequest):
    info = run_training(req.data_path, req.epochs, req.threshold, str(ART_DIR))

    # Load artifacts back into memory for serving
    by_type = json.loads((ART_DIR / "node_types.json").read_text())
    num_nodes = sum(len(by_type[t]) for t in ["card", "merchant", "device", "ip"])
    model = GraphSAGEModel(num_nodes=num_nodes)
    model.load_state_dict(torch.load(ART_DIR / "model.pt", map_location="cpu"))
    model.eval()

    df = load_transactions(req.data_path)
    indexer = build_indexer_from_df(df)
    adj, deg = build_adjacency(df, indexer)

    # Prepare cluster info again for the API
    from src.fraud_graph.train import _get_offsets
    offsets = _get_offsets(by_type)
    pairs_global, labels = transactions_to_pairs(df, indexer)
    probs = predict_scores(model, adj, deg, pairs_global)
    edges_w = []
    edges_l = []
    for (u, v), p, y in zip(pairs_global, probs, labels):
        if p >= req.threshold:
            edges_w.append((int(u), int(v), float(p)))
            edges_l.append((int(u), int(v), float(p), int(y)))
    g = build_weighted_graph(num_nodes, edges_w)
    clusters_dict = louvain_clusters(g)
    stats = compute_cluster_stats(clusters_dict, edges_l)

    STATE.update({
        "model": model,
        "adj": adj,
        "deg": deg,
        "by_type": by_type,
        "offsets": offsets,
        "threshold": req.threshold,
        "clusters": stats,
        "edges_w": edges_w,
    })

    return {"message": "trained", **info}


@app.get("/metrics")
def metrics():
    metrics_path = ART_DIR / "metrics.json"
    if metrics_path.exists():
        return JSONResponse(json.loads(metrics_path.read_text()))
    return JSONResponse({"detail": "No metrics yet"})


@app.get("/clusters")
def clusters():
    if not STATE.get("clusters"):
        return JSONResponse({"clusters": []})
    # Return top 10 clusters
    out = [c.__dict__ for c in STATE["clusters"][:10]]
    return JSONResponse({"clusters": out})


@app.get("/heatmap")
def heatmap(cluster_id: int = 0):
    if STATE.get("model") is None:
        raise HTTPException(status_code=400, detail="Model not trained")
    clusters = STATE.get("clusters", [])
    if cluster_id >= len(clusters):
        raise HTTPException(status_code=404, detail="Cluster not found")
    top = clusters[cluster_id]
    by_type = STATE["by_type"]

    # Recompute pairs/probs on the fly for clarity
    cards = by_type["card"]
    merchants = by_type["merchant"]
    # Build dense grid is unnecessary; use existing edges_w filtered by cluster nodes
    # Create local pairs and probs for heatmap
    node_set = set(top.node_ids)
    pairs_local: List[Tuple[int, int]] = []
    probs_local: List[float] = []
    for u, v, p in STATE["edges_w"]:
        if u in node_set and v in node_set:
            # Convert back to local indices in the card/merchant lists
            ci = u - STATE["offsets"]["card"]
            mi = v - STATE["offsets"]["merchant"]
            if 0 <= ci < len(cards) and 0 <= mi < len(merchants):
                pairs_local.append((ci, mi))
                probs_local.append(p)

    img = fraud_heatmap(cards, merchants, np.array(pairs_local), np.array(probs_local))
    return Response(content=img, media_type="image/png")


@app.get("/graph")
def cluster_graph(cluster_id: int = 0):
    if STATE.get("model") is None:
        raise HTTPException(status_code=400, detail="Model not trained")
    clusters = STATE.get("clusters", [])
    if cluster_id >= len(clusters):
        raise HTTPException(status_code=404, detail="Cluster not found")
    top = clusters[cluster_id]

    # Build index->(type,id)
    by_type = STATE["by_type"]
    index_to_type_id: Dict[int, Tuple[str, str]] = {}
    base = 0
    for t in ["card", "merchant", "device", "ip"]:
        for nid in by_type[t]:
            index_to_type_id[base] = (t, nid)
            base += 1
    graph_json = export_cluster_graph_json(top.node_ids, STATE["edges_w"], index_to_type_id)
    return JSONResponse(graph_json)


@app.post("/predict")
def predict(req: PredictRequest):
    if STATE.get("model") is None:
        raise HTTPException(status_code=400, detail="Model not trained")

    # Map incoming ids to indices via by_type lists
    by_type = STATE["by_type"]
    try:
        ci = by_type["card"].index(str(req.card_id))
        mi = by_type["merchant"].index(str(req.merchant_id))
    except ValueError:
        # unseen entities -> low risk fallback
        return {"transaction_id": req.transaction_id, "fraud_probability": 0.05}

    # Map to global indices
    u = STATE["offsets"]["card"] + ci
    v = STATE["offsets"]["merchant"] + mi
    pairs_global = np.array([[u, v]])
    probs = predict_scores(STATE["model"], STATE["adj"], STATE["deg"], pairs_global)
    return {"transaction_id": req.transaction_id, "fraud_probability": float(probs[0])}

