"""CLI wrapper: normalize a MatterGen output dir into a `cifs/` dir of *.cif files.

    PYTHONPATH=src python scripts/extract_cifs.py /workspace/runs/radar --prefix radar

The implementation lives in stealth.discovery.extract_cifs (so the RL loop reuses it).
"""

from __future__ import annotations

import argparse

from stealth.discovery.extract_cifs import extract

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="MatterGen output dir (contains the cif zip / extxyz)")
    ap.add_argument("--prefix", default=None, help="prefix cif names (avoids gen_0 collisions when pooling roles)")
    args = ap.parse_args()
    extract(args.run_dir, args.prefix)
