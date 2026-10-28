"""Our parser must agree with the official DatanetAPI on the same samples."""
import json
import sys
import types
from itertools import islice

import numpy as np
import pytest

from src.config import RAW_DIR, ROOT
from src.parsing import parse_tar, parse_traffic_cell, parse_result_cell

TAR = sorted(RAW_DIR.glob("*.tar.gz"))[0]
N_SAMPLES = 2


@pytest.fixture(scope="module")
def ours():
    return parse_tar(TAR, max_scenarios=N_SAMPLES, out_dir=None)


@pytest.fixture(scope="module")
def official():
    # The official reader imports jsonpickle only to decode slices_*.json, which
    # are plain JSON: a json-based stand-in gives the same objects, safely.
    sys.modules.setdefault("jsonpickle", types.SimpleNamespace(decode=json.loads))
    sys.path.insert(0, str(ROOT / "data" / "raw"))
    from datanetAPI import DatanetAPI
    api = DatanetAPI(str(RAW_DIR))
    # set_files_to_process() is unusable (its loop variable shadows the builtin
    # `tuple`, so its type check always fails): select the file directly.
    api._selected_tuple_files = [t for t in api.get_available_files() if t[1] == TAR.name]
    return list(islice(iter(api), N_SAMPLES))


def test_traffic_cell_real_rate():
    # eMBB: 1500 pkt/s, binomial {12000, 18000} bits -> 22.5 Mb/s (not the nominal 18)
    c = parse_traffic_cell("0,18000000,1500,10,2,6000,12000,18000,27")
    assert c["rate_bps"] == pytest.approx(22.5e6) and c["tos"] == 27


def test_result_cell_fields():
    r = parse_result_cell("22482.5,1498.85,242.441,0.02467,-3.70951,0.02114,0.022799,0.025115,0.026837,0.027652,7e-06")
    assert r["delay"] == pytest.approx(0.02467) and r["jitter"] == pytest.approx(7e-06)


def test_same_flows_and_targets(ours, official):
    flows = ours["flows"]
    for k, s in enumerate(official):
        perf, traffic, routing = s.get_performance_matrix(), s.get_traffic_matrix(), s.get_routing_matrix()
        sid = int(s._graph_file.split("graph_")[1].split(".")[0])
        fl = flows[flows.scenario_id == sid]
        assert len(fl) > 0
        assert s.get_global_delay() == pytest.approx(ours["scenarios"].iloc[k].global_delay)
        for _, f in fl.iterrows():
            p = perf[f.src, f.dst]["Flows"][0]
            t = traffic[f.src, f.dst]["Flows"][0]
            assert f.delay == pytest.approx(p["AvgDelay"])
            assert f.jitter == pytest.approx(p["Jitter"])
            assert f.pkts_drop == pytest.approx(p["PktsDrop"])
            assert f.pkts_gen == pytest.approx(t["PktsGen"])
            assert f.tos == t["ToS"]
            assert f.avg_pkts_lambda == pytest.approx(t["TimeDistParams"]["AvgPktsLambda"])
            # path = routing actually simulated (routing file), as in the official API
            assert list(f.path) == list(routing[f.src, f.dst])
        # every flow declared in traffic.txt is in our table (no flow forgotten)
        n_official = sum(len(traffic[i, j]["Flows"]) for i in range(s.get_network_size())
                         for j in range(s.get_network_size()))
        assert n_official == len(fl)


def test_measured_link_util(ours, official):
    links = ours["links"]
    for s in official:
        sid = int(s._graph_file.split("graph_")[1].split(".")[0])
        ps = s.get_port_stats()
        L = links[(links.scenario_id == sid) & links.meas_util.notna()]
        for _, l in L.head(50).iterrows():
            assert l.meas_util == pytest.approx(ps[l.src][l.dst]["utilization"])


def test_no_impossible_values(ours):
    f = ours["flows"]
    assert (f.pkts_gen > 0).all()
    assert (f.pkts_drop >= 0).all() and (f.pkts_drop <= f.pkts_gen * 1.001).all()
    assert (f.path_len >= 1).all()
    assert np.isfinite(f.rate_bps).all()
