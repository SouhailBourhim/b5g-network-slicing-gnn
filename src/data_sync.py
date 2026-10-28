"""Share the processed tables so the whole team works on bit-identical data.

The processed parquet files (~270 MB) are published as assets of a GitHub release
(tag data-vN) with a SHA256SUMS file. Teammates download and verify them instead of
re-parsing the raw archives. Requires the GitHub CLI (`gh auth login`) and read
access to the (private) repository.

  python -m src.data_sync fetch                 # download + verify data-v1 (everyone)
  python -m src.data_sync verify                # re-check local files against SHA256SUMS
  python -m src.data_sync publish --tag data-v1 # upload (data owner only)
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys

from src.config import PROCESSED_DIR

REPO = "SouhailBourhim/b5g-network-slicing-gnn"
DEFAULT_TAG = "data-v1"
FILES = ["scenarios", "links", "flows", "flow_links",
         "flow_features", "hop_features", "link_features", "splits"]
SUMS = PROCESSED_DIR / "SHA256SUMS"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gh(*args):
    print("$ gh", " ".join(args))
    subprocess.run(["gh", *args], check=True)


def publish(tag: str):
    paths = [PROCESSED_DIR / f"{name}.parquet" for name in FILES]
    SUMS.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in paths), encoding="utf-8")
    notes = ("Processed tables of the Farreras et al. (2024) slicing dataset (Zenodo 10.5281/zenodo.10610616, "
             "CC BY 4.0), built by src/parsing.py + src/features.py + src/splits.py. "
             "Fetch with: python -m src.data_sync fetch --tag " + tag)
    gh("release", "create", tag, "--repo", REPO, "--title", f"Processed data {tag}", "--notes", notes,
       *[str(p) for p in paths], str(SUMS))


def verify() -> bool:
    ok = True
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        p = PROCESSED_DIR / name
        good = p.exists() and sha256(p) == digest
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name}")
    return ok


def fetch(tag: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gh("release", "download", tag, "--repo", REPO, "--dir", str(PROCESSED_DIR), "--clobber")
    if not verify():
        sys.exit("checksum mismatch: delete the BAD files and fetch again")
    print(f"data {tag} ready in {PROCESSED_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fetch", "verify", "publish"])
    ap.add_argument("--tag", default=DEFAULT_TAG)
    a = ap.parse_args()
    {"fetch": lambda: fetch(a.tag), "verify": lambda: sys.exit(0 if verify() else 1),
     "publish": lambda: publish(a.tag)}[a.action]()


if __name__ == "__main__":
    main()
