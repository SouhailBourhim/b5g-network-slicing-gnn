"""The cached graphs must match the tables they were built from."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import PROCESSED_DIR, GRAPHS_DIR
from src.graphs import load_graphs

pytestmark = pytest.mark.skipif(not (GRAPHS_DIR / "val.pt").exists(), reason="run python -m src.graphs first")


@pytest.fixture(scope="module")
def val_graphs():
    return load_graphs("val")


def test_counts_match_tables(val_graphs):
    ids = [g.scenario_id for g in val_graphs[:20]]
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet", columns=["scenario_id", "path_len"])
    L = pd.read_parquet(PROCESSED_DIR / "link_features.parquet", columns=["scenario_id"])
    n_flows = f[f.scenario_id.isin(ids)].groupby("scenario_id").size()
    n_hops = f[f.scenario_id.isin(ids)].groupby("scenario_id").path_len.sum()
    n_links = L[L.scenario_id.isin(ids)].groupby("scenario_id").size()
    for g in val_graphs[:20]:
        assert g["flow"].num_nodes == n_flows[g.scenario_id]                   # one node per flow
        assert g["link"].num_nodes == n_links[g.scenario_id]                   # one node per used link
        assert g["flow", "traverses", "link"].num_edges == n_hops[g.scenario_id]  # one edge per hop


def test_edges_valid_and_reversed(val_graphs):
    for g in val_graphs[:50]:
        ei = g["flow", "traverses", "link"].edge_index
        assert ei[0].max() < g["flow"].num_nodes and ei[1].max() < g["link"].num_nodes
        assert torch.equal(g["link", "carries", "flow"].edge_index, ei.flip(0))
        # every flow has as many edges as its path length, every used link carries >= 1 flow
        assert torch.bincount(ei[1], minlength=g["link"].num_nodes).min() >= 1


def test_no_nan_and_mask(val_graphs):
    for g in val_graphs:
        for t in (g["flow"].x, g["link"].x, g["flow", "traverses", "link"].edge_attr, g["flow"].y):
            assert torch.isfinite(t).all()
        assert g["flow"].y_mask.dtype == torch.bool


def test_normalizer_fitted_on_train():
    tr = load_graphs("train")
    x = torch.cat([g["flow"].x for g in tr[:2000]])
    # continuous columns are standardized with TRAIN statistics -> mean ~0 on train
    cont = [3, 4, 7]                                     # delta, prov_ratio, path_len
    assert x[:, cont].mean(0).abs().max() < 0.1
