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
from .optical_bridge import candidate_material, load_gnnopt_nk, material_from_gnnopt
from .screen import demo_structures, load_cifs, predict, screen

OUTPUT = REPO_ROOT / "data" / "discovery_to_physics.parquet"


def run(cif_dir=None, role="dielectric_spacer", thickness_um=2.0, gnnopt_nk=None) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    structures = load_cifs(cif_dir) if cif_dir else demo_structures()
    ranked = screen(predict(structures), role)

    nk = load_gnnopt_nk(gnnopt_nk) if gnnopt_nk else {}
    ground = optics.Layer("Aluminum (ground plane)", 0.3)
    rows = []
    for _, r in ranked.iterrows():
        if r["id"] in nk:
            mat = material_from_gnnopt(r["id"], nk[r["id"]], r["best_role"])
            source = "GNNOpt"
        else:
            mat = candidate_material(r["id"], float(r["band_gap_ev"]), r["best_role"])
            source = "estimate"
        # real visible n,k (where GNNOpt is trustworthy)
        n_vis = complex(optics.optical_constants(mat, 0.55, clip=True)[0])
        stack = optics.Stack.of(optics.Layer(mat, thickness_um), ground)
        rows.append(
            {
                "id": r["id"],
                "formula": r["formula"],
                "band_gap_ev": round(float(r["band_gap_ev"]), 3),
                "best_role": r["best_role"],
                "nk_source": source,
                "n_550nm": round(n_vis.real, 3),
                "k_550nm": round(n_vis.imag, 3),
                "lwir_emissivity": round(optics.band_emissivity(stack, (8.0, 14.0), clip=True), 3),
                "mwir_emissivity": round(optics.band_emissivity(stack, (3.0, 5.0), clip=True), 3),
            }
        )
    return pd.DataFrame(rows)


def main(cif_dir=None, gnnopt_nk=None) -> None:
    df = run(cif_dir, gnnopt_nk=gnnopt_nk)
    pd.set_option("display.width", 180)
    src = "GNNOpt n,k" if gnnopt_nk else "band-gap-estimated n,k"
    print(f"Generated material -> {src} -> TMM physics (stack: material(2um)/Al):\n")
    print(df.to_string(index=False))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    n_gnn = int((df["nk_source"] == "GNNOpt").sum())
    print(f"\n-> {OUTPUT}   ({n_gnn}/{len(df)} used GNNOpt n,k)")
    print(
        "Visible/NIR n,k are GNNOpt-predicted (real electronic optics); LWIR stays approximate."
        if gnnopt_nk
        else "(n,k band-gap-estimated; pass --gnnopt-nk <json> for real visible/NIR optics.)"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", default=None, help="dir of MatterGen CIFs (default: demo structures)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--gnnopt-nk", default=None, help="GNNOpt n,k JSON for real visible/NIR optics")
    args = ap.parse_args()
    main(cif_dir=None if args.demo else args.cif_dir, gnnopt_nk=args.gnnopt_nk)
