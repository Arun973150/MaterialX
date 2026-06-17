"""End-to-end discovery PoC: generate(materials) -> screen -> n,k bridge -> physics.

Demonstrates a (generated or candidate) material flowing all the way into the TMM
physics: predict its properties, estimate its n,k from the band gap, place it over a
ground plane, and *compute* its LWIR emissivity. This is the closed loop the gist asks
for — novel material in, simulated signature out.

    python -m stealth.discovery.pipeline --demo
    python -m stealth.discovery.pipeline --cif-dir runs/dielectric    # RunPod: MatterGen output
"""

from __future__ import annotations

import warnings

import pandas as pd

from ..config import REPO_ROOT
from ..physics import optics
from .optical_bridge import candidate_material
from .screen import demo_structures, load_cifs, predict, screen

OUTPUT = REPO_ROOT / "data" / "discovery_to_physics.parquet"


def run(cif_dir: str | None = None, role: str = "dielectric_spacer", thickness_um: float = 2.0) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    structures = load_cifs(cif_dir) if cif_dir else demo_structures()
    ranked = screen(predict(structures), role)

    ground = optics.Layer("Aluminum (ground plane)", 0.3)
    rows = []
    for _, r in ranked.iterrows():
        mat = candidate_material(r["id"], float(r["band_gap_ev"]), r["best_role"])
        stack = optics.Stack.of(optics.Layer(mat, thickness_um), ground)
        e_lwir = optics.band_emissivity(stack, (8.0, 14.0), clip=True)
        e_mwir = optics.band_emissivity(stack, (3.0, 5.0), clip=True)
        rows.append(
            {
                "id": r["id"],
                "formula": r["formula"],
                "band_gap_ev": round(float(r["band_gap_ev"]), 3),
                "best_role": r["best_role"],
                "lwir_emissivity": round(e_lwir, 3),
                "mwir_emissivity": round(e_mwir, 3),
                "ir_stealth_ok": e_lwir < 0.3,
            }
        )
    return pd.DataFrame(rows)


def main(cif_dir: str | None = None) -> None:
    df = run(cif_dir)
    pd.set_option("display.width", 160)
    print("Generated/candidate material -> estimated n,k -> TMM physics (stack: material(2um)/Al):\n")
    print(df.to_string(index=False))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"\nClosed loop: {int(df['ir_stealth_ok'].sum())}/{len(df)} read as low-IR-emissivity. -> {OUTPUT}")
    print("(n,k estimated from predicted band gap — first-order; trained optical GNN would refine.)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", default=None, help="dir of MatterGen CIFs (default: demo structures)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    main(cif_dir=None if args.demo else args.cif_dir)
