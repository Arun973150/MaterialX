"""#2 generator specialization — build a RAM-specialized fine-tuning set for MatterGen.

The gist's step 2 is "train a generative NN on the molecular structures of existing radar-absorbent
and metamaterial coatings." Pretrained MatterGen samples generic inorganics; this assembles the
*stealth-relevant* training set so its adapter can be fine-tuned (RAFT/property-conditioned) toward
real absorber chemistry.

From the #1 Tier-A1 table (`dataset.py` -> em_dataset.parquet), filter to each role's RAM chemistry
family (manufacturability.practical_elements) and attach the role's MatterGen conditioning property
(magmom for magnetic absorbers, band gap for conductors/dielectrics). Writes per-role CIFs + a
labels CSV that the MatterGen fine-tuning data module ingests.

    python -m stealth.discovery.finetune_dataset --em data/em_dataset.parquet --role radar_magnetic \
        --out data/finetune/radar_magnetic
Then fine-tune on the pod (see RUNPOD section 7) and generate with the fine-tuned checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from .manufacturability import practical_elements
from .targets import get_target

# role -> the em_dataset column used as the conditioning label
_LABEL_COL = {
    "radar_magnetic": "magmom",
    "radar_conductor": "bandgap_optb88",
    "ir_phasechange": "bandgap_optb88",
    "dielectric_spacer": "bandgap_optb88",
}


def select_for_role(df, role: str, max_other_frac: float = 0.0):
    """Filter a #1 dataset DataFrame to a role's RAM chemistry family + a non-null label.

    Keeps rows whose elements are within the role's practical family (allowing up to
    `max_other_frac` of out-of-family elements by count) and that have the conditioning label.
    """
    from pymatgen.core import Composition

    allowed = set(practical_elements(role))
    label_col = _LABEL_COL[role]
    keep = []
    for _, r in df.iterrows():
        if r.get(label_col) is None:
            continue
        try:
            els = [e.symbol for e in Composition(r["formula"]).elements]
        except Exception:  # noqa: BLE001
            continue
        out_frac = sum(1 for e in els if e not in allowed) / max(1, len(els))
        if out_frac <= max_other_frac:
            keep.append(r.name)
    return df.loc[keep].reset_index(drop=True)


def build_finetune_dataset(em_parquet: str, role: str, out_dir: str) -> int:
    """Write per-role CIFs + labels.csv for MatterGen fine-tuning. Returns the count."""
    import pandas as pd

    from .dataset import to_structure

    df = pd.read_parquet(em_parquet)
    sel = select_for_role(df, role)
    out = Path(out_dir)
    (out / "structures").mkdir(parents=True, exist_ok=True)

    label_col = _LABEL_COL[role]
    prop = get_target(role).mattergen_property
    rows = []
    for _, r in sel.iterrows():
        try:
            struct = to_structure(r["atoms"])
        except Exception:  # noqa: BLE001
            continue
        cif_path = out / "structures" / f"{r['jid']}.cif"
        struct.to(filename=str(cif_path))
        rows.append({"material_id": r["jid"], "formula": r["formula"],
                     "cif": f"structures/{r['jid']}.cif", prop: float(r[label_col])})

    labels = pd.DataFrame(rows)
    labels.to_csv(out / "labels.csv", index=False)
    print(f"[{role}] {len(labels)} RAM-family structures -> {out}/labels.csv "
          f"(conditioning property: {prop})")
    return len(labels)


def main(em_parquet: str, role: str, out_dir: str) -> None:
    n = build_finetune_dataset(em_parquet, role, out_dir)
    print(f"\nNext (pod): fine-tune MatterGen's adapter on this set, conditioning on "
          f"'{get_target(role).mattergen_property}', then generate with the fine-tuned checkpoint. "
          f"See RUNPOD section 7. ({n} structures)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--em", default="data/em_dataset.parquet", help="#1 dataset parquet")
    ap.add_argument("--role", required=True, choices=list(_LABEL_COL))
    ap.add_argument("--out", required=True, help="output dir for CIFs + labels.csv")
    args = ap.parse_args()
    main(args.em, args.role, args.out)
