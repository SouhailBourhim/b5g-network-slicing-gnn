"""Share the processed tables so the whole team works on bit-identical data.

The processed parquet files (~270 MB) are published as assets of a GitHub release
(tag data-vN) with a SHA256SUMS file. Teammates download and verify them instead of
re-parsing the raw archives. The repository is public: `fetch` is a plain HTTPS
download (no account needed). Only `publish` needs the GitHub CLI (`gh auth login`).

  python -m src.data_sync fetch                 # download + verify data-v1 (everyone)
  python -m src.data_sync verify                # re-check local files against SHA256SUMS
  python -m src.data_sync publish --tag data-v1 # upload (data owner only)
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

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


def verify(dest: Path = PROCESSED_DIR) -> bool:
    ok = True
    for line in (dest / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        p = dest / name
        good = p.exists() and sha256(p) == digest
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name}")
    return ok


def download(url: str, target: Path):
    """Stream a release asset to disk (large files: never held in memory)."""
    with urllib.request.urlopen(url) as r, open(target, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def fetch(tag: str, dest: Path = PROCESSED_DIR, only: list[str] | None = None):
    """Download SHA256SUMS then every parquet that is missing or fails its checksum."""
    dest.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/{REPO}/releases/download/{tag}"
    download(f"{base}/SHA256SUMS", dest / "SHA256SUMS")
    for line in (dest / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        if only and name not in only:
            continue
        p = dest / name
        if p.exists() and sha256(p) == digest:
            print(f"already up to date: {name}")
            continue
        print(f"downloading {name} ...")
        download(f"{base}/{name}", p)
        if sha256(p) != digest:
            sys.exit(f"checksum mismatch for {name}: delete it and fetch again")
    print(f"data {tag} ready in {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fetch", "verify", "publish"])
    ap.add_argument("--tag", default=DEFAULT_TAG)
    a = ap.parse_args()
    {"fetch": lambda: fetch(a.tag), "verify": lambda: sys.exit(0 if verify() else 1),
     "publish": lambda: publish(a.tag)}[a.action]()


if __name__ == "__main__":
    main()
