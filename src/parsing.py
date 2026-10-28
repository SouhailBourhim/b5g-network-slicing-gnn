"""Parse the Farreras et al. (2024) network-slicing dataset into tidy tables.

Why our own parser instead of DatanetAPI.__iter__ (see JOURNAL.md D0.2):
  * every sample stores N x N matrices (N ~ 400 nodes -> ~170k cells) but only
    the ~400 cells holding a real flow matter. We only touch those cells.
  * slices_*.json is plain JSON; the official API uses jsonpickle.decode, which
    can instantiate arbitrary Python objects. We use json.

Raw layout (one line per scenario in every file of a tarball, same order):
  input_files.txt        "<k>;graph_K.txt;routing_K.txt;slices_K.json;"
  stability.txt          "<sim_time>;<conv>;OK;Mem(KB):..;Time(s):.."
  traffic.txt            "<maxAvgLambda>|cell;cell;..."   cell = generator params + ToS
  simulationResults.txt  "<gPkts>,<gLoss>,<gDelay>|cell;cell;..."  cell = 11 measures
  linkUsage.txt          "cell;cell;..."  cell = port stats ("-1" if no link)
Cell index for the pair (src, dst) is src * N + dst.

Output: four tables (one row per scenario / directed link / flow / flow-hop).
Usage:
  python -m src.parsing --max-tars 1 --workers 1        # quick test
  python -m src.parsing --workers 8                     # full dataset
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

from src.config import RAW_DIR, PARTS_DIR, PROCESSED_DIR, LOGS_DIR

# Order of the 11 numbers in a simulationResults flow cell (see datanetAPI.py).
RESULT_FIELDS = ["avg_bw_kbps", "pkts_gen", "pkts_drop", "delay", "ln_delay",
                 "p10", "p20", "p50", "p80", "p90", "jitter"]
CORE_TYPES = {"inner", "antenna"}  # Topology Zoo routers + radio access points


# --------------------------------------------------------------------------
# Small parsers for one cell / one attribute
# --------------------------------------------------------------------------
def parse_traffic_cell(cell: str) -> dict:
    """Decode the generator parameters of one flow (traffic.txt cell).

    Format (datanetAPI._timedistparams/_sizedistparams):
      time dist 0 (Poisson): 0,EqLambda,AvgPktsLambda,ExpMaxFactor, <size>, ToS
      time dist 1 (CBR):     1,EqLambda,AvgPktsLambda,              <size>, ToS
      size dist 0: 0,AvgPktSize        size dist 2 (binomial): 2,Avg,Size1,Size2
    """
    v = cell.split(",")
    out = {"time_dist": int(v[0]), "eq_lambda": float(v[1]), "avg_pkts_lambda": float(v[2])}
    if out["time_dist"] == 0:
        out["exp_max_factor"] = float(v[3])
        k = 4
    elif out["time_dist"] == 1:
        out["exp_max_factor"] = float("nan")
        k = 3
    else:
        raise ValueError(f"unsupported time distribution in cell {cell[:60]}")
    out["size_dist"] = int(v[k])
    out["avg_pkt_size_param"] = float(v[k + 1])
    if out["size_dist"] == 2:            # binomial: two possible packet sizes
        out["pkt_size_1"], out["pkt_size_2"] = float(v[k + 2]), float(v[k + 3])
        # Mean packet size actually generated (p = 0.5, checked against the
        # measured AvgBw / PktsGen in the EDA). See JOURNAL D0.3.
        out["mean_pkt_size"] = (out["pkt_size_1"] + out["pkt_size_2"]) / 2
    elif out["size_dist"] == 0:          # deterministic size
        out["pkt_size_1"] = out["pkt_size_2"] = out["avg_pkt_size_param"]
        out["mean_pkt_size"] = out["avg_pkt_size_param"]
    else:
        raise ValueError(f"unsupported size distribution in cell {cell[:60]}")
    out["tos"] = int(v[-1])
    # Offered load computed from inputs only (packets/s x bits/packet).
    out["rate_bps"] = out["avg_pkts_lambda"] * out["mean_pkt_size"]
    return out


def parse_result_cell(cell: str) -> dict:
    """Decode the 11 measured values of one flow (simulationResults.txt cell)."""
    if ":" in cell:
        raise ValueError("more than one flow on a src-dst pair (not expected here)")
    return dict(zip(RESULT_FIELDS, map(float, cell.split(","))))


def parse_link_usage_cell(cell: str) -> dict:
    """Port-level measured stats: utilization, loss ratio, mean packet size.

    These are SIMULATION OUTPUTS (prefixed meas_): never used as model inputs.
    """
    port = cell.split(":")[0].split(",")
    to_f = lambda x: float("nan") if "nan" in x else float(x)
    return {"meas_util": to_f(port[0]), "meas_loss": to_f(port[1]),
            "meas_avg_pkt_size": to_f(port[2]), "n_queues_meas": cell.count(":")}


def read_routing_file(routing_file: Path) -> list[list[int]]:
    """Read a destination-based routing file: R[node][dst] = output port.

    Each line i lists, for every destination j, the port node i uses to reach j
    (-1 = node i is the destination). Same format as DatanetAPI._readRoutingFile.
    """
    with open(routing_file, "r") as f:
        first = f.readline()
        if ";" in first:
            raise ValueError(f"{routing_file.name}: path-list routing format not handled")
        lines = [first] + f.readlines()
    return [[int(float(p)) for p in line.split(",")[:-1]] for line in lines]


def routing_path(R: list[list[int]], port_to_next: dict, src: int, dst: int) -> list[int]:
    """Follow the output ports hop by hop from src until dst is reached."""
    path, node = [src], src
    while R[node][dst] != -1:
        node = port_to_next[node][R[node][dst]]
        path.append(node)
        if len(path) > len(R):                 # safety net against routing loops
            raise ValueError(f"routing loop {src}->{dst}")
    return path


def node_queue_config(node: dict) -> tuple[dict, list, list]:
    """WFQ configuration of a node.

    Returns
      tos_to_queue : {ToS -> queue index}
      weights      : list (one per output port) of lists (one weight per queue)
      queue_sizes  : list of queue sizes in packets
    'schedulingWeights' == '-' means a single FIFO queue (leaf nodes).
    """
    tos_to_queue = {}
    for q, tos_list in enumerate(str(node.get("tosToQoSqueue", "")).split(";")):
        for t in tos_list.split(","):
            if t != "":
                tos_to_queue[int(t)] = q
    w_raw = str(node.get("schedulingWeights", "-"))
    weights = [] if w_raw in ("-", "") else [[float(x) for x in w.split(",")] for w in w_raw.split(";")]
    queue_sizes = [int(x) for x in str(node.get("queueSizes", "32")).split(",")]
    return tos_to_queue, weights, queue_sizes


# --------------------------------------------------------------------------
# One scenario
# --------------------------------------------------------------------------
def parse_scenario(root: Path, lines: dict) -> dict[str, list[dict]]:
    """Turn the 5 raw lines of one scenario into rows of the 4 tables."""
    files = lines["input"].split(";")
    scenario_id = int(files[1].replace("graph_", "").replace(".txt", ""))
    G = nx.read_gml(root / "graphs" / files[1], destringizer=int)
    with open(root / "slices" / files[3], "r") as f:
        slices = json.load(f)
    N = G.number_of_nodes()

    head_t, t_cells = lines["traffic"].split("|", 1)
    head_r, r_cells = lines["results"].split("|", 1)
    t_cells, r_cells = t_cells.split(";"), r_cells.split(";")
    u_cells = lines["link_usage"].split(";")
    g_pkts, g_loss, g_delay = map(float, head_r.split(","))
    status = lines["status"].split(";")

    # ---- links: every directed edge of the topology ----
    node_type = {n: G.nodes[n]["type"] for n in G.nodes}
    link_rows, link_id = [], {}
    for (u, v, _k, d) in G.edges(keys=True, data=True):
        lid = len(link_rows)
        link_id[(u, v)] = lid
        row = {"scenario_id": scenario_id, "link_id": lid, "src": u, "dst": v,
               "src_type": node_type[u], "dst_type": node_type[v],
               "capacity_bps": float(d["bandwidth"]), "port": int(d["port"]),
               "sched_policy": G.nodes[u].get("schedulingPolicy", "FIFO"),
               "levels_qos": int(G.nodes[u].get("levelsQoS", 1))}
        cell = u_cells[u * N + v]
        row.update(parse_link_usage_cell(cell) if cell != "-1" else
                   {"meas_util": float("nan"), "meas_loss": float("nan"),
                    "meas_avg_pkt_size": float("nan"), "n_queues_meas": 0})
        link_rows.append(row)

    # ---- routing actually simulated (routing_K.txt), see JOURNAL D1.1 ----
    # The 'path' field of the slices file differs from the simulated routing in
    # many scenarios; the measured link utilization matches the routing file.
    R = read_routing_file(root / "routings" / files[2])
    port_to_next = {u: {} for u in G.nodes}
    for (u, v, _k, d) in G.edges(keys=True, data=True):
        port_to_next[u][int(d["port"])] = v

    # ---- flows (declared in the slices file) + their hops ----
    qconf = {n: node_queue_config(G.nodes[n]) for n in G.nodes}
    flow_rows, hop_rows = [], []
    for s in slices:
        for f in s["flows"]:
            src, dst = int(f["origin_node"]), int(f["destination"])
            slice_path = json.loads(f["path"]) if isinstance(f["path"], str) else f["path"]
            path = routing_path(R, port_to_next, src, dst)
            fid = len(flow_rows)
            row = {"scenario_id": scenario_id, "flow_id": fid, "src": src, "dst": dst,
                   "antenna": int(f["origin_node_antenna"]), "slice_type": s["type"],
                   "slice_num": int(s["number"]), "delta": float(s["delta"]),
                   "nominal_bw": float(f["bandwidth"]), "path_len": len(path) - 1,
                   "path": path, "slice_path_len": len(slice_path) - 1,
                   "path_matches_slice_file": path == slice_path}
            row.update(parse_traffic_cell(t_cells[src * N + dst]))
            row.update(parse_result_cell(r_cells[src * N + dst]))
            flow_rows.append(row)

            for hop, (a, b) in enumerate(zip(path[:-1], path[1:])):
                tos_to_q, weights, qsizes = qconf[a]
                port = G[a][b][0]["port"]
                q = tos_to_q.get(row["tos"], len(qsizes) - 1)
                if weights and port < len(weights):
                    w = weights[port]
                    share = w[q] / sum(w)              # WFQ guaranteed share
                else:
                    share = 1.0                        # single FIFO queue
                hop_rows.append({"scenario_id": scenario_id, "flow_id": fid, "hop": hop,
                                 "link_id": link_id[(a, b)], "queue": q,
                                 "is_default_queue": q == len(qsizes) - 1 and len(qsizes) > 1,
                                 "wfq_share": share,
                                 "queue_size_pkts": qsizes[min(q, len(qsizes) - 1)]})

    core = [n for n in G.nodes if node_type[n] in CORE_TYPES]
    scen = {"scenario_id": scenario_id, "tar_file": lines["tar"],
            "topology": G.graph.get("label", G.graph.get("Network", "unknown")),
            "n_nodes": N, "n_core_nodes": len(core),
            "n_core_links": sum(1 for (u, v) in link_id if u in core and v in core),
            "n_links": len(link_rows), "n_flows": len(flow_rows), "n_slices": len(slices),
            "max_avg_lambda": float(head_t), "global_packets": g_pkts,
            "global_losses": g_loss, "global_delay": g_delay,
            "sim_time": float(status[0]), "sim_conv": float(status[1]),
            "sim_mem_kb": int(status[3].split(":")[1]),
            "sim_wallclock_s": int(status[4].split(":")[1]) if len(status) > 4 else -1}
    for t in ("eMBB", "URLLC", "mMTC"):
        scen[f"n_slices_{t}"] = sum(s["type"] == t for s in slices)
    return {"scenarios": [scen], "links": link_rows, "flows": flow_rows, "flow_links": hop_rows}


# --------------------------------------------------------------------------
# One tarball (~100 scenarios), streamed line by line
# --------------------------------------------------------------------------
def iter_tar_lines(tar_path: Path):
    """Yield the 5 aligned raw lines of each scenario of a tarball."""
    with tarfile.open(tar_path, "r:gz") as tar:
        d = tar.next().name
        names = {"input": "input_files.txt", "status": "stability.txt",
                 "traffic": "traffic.txt", "results": "simulationResults.txt",
                 "link_usage": "linkUsage.txt"}
        fh = {k: tar.extractfile(f"{d}/{v}") for k, v in names.items()}
        while True:
            lines = {k: f.readline().decode().rstrip("\r\n") for k, f in fh.items()}
            if not lines["results"] or not lines["traffic"]:
                break
            lines["tar"] = tar_path.name
            yield lines


def parse_tar(tar_path: Path, root: Path = RAW_DIR, max_scenarios: int | None = None,
              out_dir: Path | None = PARTS_DIR) -> dict[str, pd.DataFrame]:
    """Parse one tarball; skip scenarios whose simulation did not reach 'OK'."""
    tables = {k: [] for k in ("scenarios", "links", "flows", "flow_links")}
    n_skipped = 0
    for i, lines in enumerate(iter_tar_lines(tar_path)):
        if max_scenarios is not None and i >= max_scenarios:
            break
        if ";OK;" not in lines["status"]:      # same filter as the official API
            n_skipped += 1
            continue
        for k, rows in parse_scenario(root, lines).items():
            tables[k].extend(rows)
    dfs = {k: pd.DataFrame(v) for k, v in tables.items()}
    dfs["scenarios"]["n_skipped_in_tar"] = n_skipped
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, df in dfs.items():
            df.to_parquet(out_dir / f"{k}__{tar_path.stem.replace('.tar', '')}.parquet", index=False)
    return dfs


def _worker(args):
    """Parse one tarball; never raise, so one bad archive cannot kill the pool."""
    tar_path, max_scenarios = args
    t0 = time.time()
    try:
        dfs = parse_tar(tar_path, max_scenarios=max_scenarios)
        return tar_path.name, len(dfs["scenarios"]), len(dfs["flows"]), time.time() - t0, None
    except BaseException:
        return tar_path.name, 0, 0, time.time() - t0, traceback.format_exc()


def merge_parts(parts_dir: Path = PARTS_DIR, out_dir: Path = PROCESSED_DIR) -> None:
    """Concatenate the per-tar parquet files into one file per table."""
    for k in ("scenarios", "links", "flows", "flow_links"):
        files = sorted(parts_dir.glob(f"{k}__*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.to_parquet(out_dir / f"{k}.parquet", index=False)
        print(f"{k:11s}: {len(df):>10,d} rows <- {len(files)} parts")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tars", type=int, default=None, help="only the first K tarballs")
    ap.add_argument("--max-scenarios", type=int, default=None, help="per tarball (testing)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true", help="resume an interrupted run")
    args = ap.parse_args()

    tars = sorted(RAW_DIR.glob("*.tar.gz"))[: args.max_tars]
    if args.skip_existing:
        done = {p.name.split("__", 1)[1] for p in PARTS_DIR.glob("flows__*.parquet")}
        tars = [t for t in tars if t.name.replace(".tar.gz", ".parquet") not in done]
    LOGS_DIR.mkdir(exist_ok=True)
    log = open(LOGS_DIR / "parsing.log", "a", encoding="utf-8", buffering=1)

    def say(msg: str):
        tqdm.write(msg, file=sys.stdout)
        log.write(msg + "\n")

    say(f"Parsing {len(tars)} tarballs with {args.workers} workers -> {PARTS_DIR}")
    t0, failed = time.time(), []
    with Pool(args.workers) as pool:
        for name, n_s, n_f, dt, err in tqdm(pool.imap_unordered(_worker, [(t, args.max_scenarios) for t in tars]),
                                            total=len(tars), unit="tar", file=sys.stdout):
            if err:
                failed.append(name)
                say(f"  FAILED {name} after {dt:.0f}s:\n{err}")
            else:
                say(f"  {name}: {n_s} scenarios, {n_f} flows in {dt:.0f}s")
    if failed:
        say(f"{len(failed)} archive(s) failed: {failed}. Fix, then rerun with --skip-existing.")
        sys.exit(1)
    merge_parts()
    say(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
