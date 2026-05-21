from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import pandas as pd

from ouroboros.config import LABEL_SUCCESS

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    strategy_label: str
    cluster_id: int
    cluster_name: str


@functools.lru_cache(maxsize=1)
def _load_sbert():
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2) …")
    return SentenceTransformer("all-MiniLM-L6-v2")


def cluster_strategies(labels: list[str]) -> list[ClusterAssignment]:
    """Embed strategy labels and cluster via HDBSCAN. Returns one assignment per label."""
    if len(labels) < 3:
        return [ClusterAssignment(l, -1, "unclustered") for l in labels]

    import numpy as np
    import hdbscan  # type: ignore[import]

    model = _load_sbert()
    embeddings = model.encode(labels, show_progress_bar=False)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, allow_single_cluster=True)
    cluster_ids = clusterer.fit_predict(embeddings)

    # Label each cluster with its medoid (the example closest to cluster centroid)
    unique_clusters = set(cluster_ids)
    cluster_names: dict[int, str] = {-1: "unclustered"}
    for cid in unique_clusters:
        if cid == -1:
            continue
        idxs = [i for i, c in enumerate(cluster_ids) if c == cid]
        centroid = embeddings[idxs].mean(axis=0)
        medoid_idx = min(idxs, key=lambda i: float(np.linalg.norm(embeddings[i] - centroid)))
        cluster_names[cid] = labels[medoid_idx]

    return [
        ClusterAssignment(
            strategy_label=lbl,
            cluster_id=int(cid),
            cluster_name=cluster_names.get(int(cid), "unclustered"),
        )
        for lbl, cid in zip(labels, cluster_ids)
    ]


def cluster_success_rate(run_df: pd.DataFrame, assignments: list[ClusterAssignment]) -> pd.DataFrame:
    """Compute E(s): success rate per strategy cluster."""
    if run_df.empty or not assignments:
        return pd.DataFrame(columns=["cluster_id", "cluster_name", "n_attempts", "n_success", "success_rate"])

    assign_df = pd.DataFrame([
        {"strategy_label": a.strategy_label, "cluster_id": a.cluster_id, "cluster_name": a.cluster_name}
        for a in assignments
    ])

    merged = run_df[run_df["outcome"] == LABEL_SUCCESS].merge(
        assign_df, on="strategy_label", how="left"
    )
    all_attempts = run_df.merge(assign_df, on="strategy_label", how="left")

    rows = []
    for (cid, cname), grp in all_attempts.groupby(["cluster_id", "cluster_name"]):
        n_att = len(grp)
        n_suc = (grp["outcome"] == LABEL_SUCCESS).sum()
        rows.append({
            "cluster_id": int(cid),
            "cluster_name": cname,
            "n_attempts": n_att,
            "n_success": int(n_suc),
            "success_rate": round(n_suc / n_att, 4) if n_att else 0.0,
        })
    return pd.DataFrame(rows).sort_values("success_rate", ascending=False)
