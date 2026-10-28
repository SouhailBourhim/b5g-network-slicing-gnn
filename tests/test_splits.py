"""The split must be leak-free and reproducible."""
import pandas as pd
import pytest

from src.config import PROCESSED_DIR
from src.splits import make_splits, load_splits


@pytest.fixture(scope="module")
def splits():
    return make_splits()


def test_one_split_per_scenario(splits):
    assert splits.scenario_id.is_unique                       # a scenario lives in exactly one split


def test_heldout_topologies_never_seen(splits):
    held = set(splits[splits.split == "test_unseen_topo"].topology)
    seen = set(splits[splits.split != "test_unseen_topo"].topology)
    assert held and not (held & seen)                         # no topology on both sides


def test_fractions(splits):
    pool = splits[splits.split != "test_unseen_topo"].split.value_counts(normalize=True)
    assert pool["train"] == pytest.approx(0.70, abs=0.02)
    assert pool["val"] == pytest.approx(0.15, abs=0.02)
    assert pool["test"] == pytest.approx(0.15, abs=0.02)


def test_deterministic(splits):
    again = make_splits()
    pd.testing.assert_frame_equal(splits.reset_index(drop=True), again.reset_index(drop=True))


def test_every_slice_in_every_split(splits):
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet", columns=["scenario_id", "slice_type"])
    tab = f.merge(splits, on="scenario_id").groupby(["split", "slice_type"]).size().unstack()
    assert (tab > 0).all().all()


def test_frozen_file_matches_code(splits):
    # data/processed/splits.parquet is what models load: it must equal what the code produces
    frozen = load_splits().sort_values("scenario_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(frozen, splits.sort_values("scenario_id").reset_index(drop=True))
