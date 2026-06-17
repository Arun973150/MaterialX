"""S.U.N. validation of generated materials — Stable, Unique, Novel.

The field-standard quality metric for a generative materials model (it's how MatterGen
itself is evaluated). For each generated structure:

  * Stable  — relax with an MLIP (matgl M3GNet potential) and compute the energy above
              the Materials Project convex hull; stable if E_hull < 0.1 eV/atom.
  * Unique  — not a structural duplicate of another generated structure (StructureMatcher).
  * Novel   — its reduced composition is not already in Materials Project.

Uses only tools already in the base env (matgl + pymatgen + mp-api) — no new environment.

Note on E_hull: the candidate's energy comes from the MLIP while the hull is MP's DFT
energies. matgl's potential is trained to reproduce DFT, so this is a standard *fast-screening*
approximation; the exact value would come from consistent-reference DFT (the deferred step).

    python -m stealth.discovery.stability --cif-dir /workspace/runs/radar/cifs
"""

from __future__ import annotations

import warnings

import pandas as pd

from ..config import REPO_ROOT
from .screen import demo_structures, load_cifs

OUTPUT = REPO_ROOT / "data" / "discovery_sun.parquet"
REPORT = REPO_ROOT / "reports" / "discovery_sun.md"

EHULL_STABLE = 0.1  # eV/atom

_RELAXER = None


def _relaxer():
    global _RELAXER
    if _RELAXER is None:
        warnings.filterwarnings("ignore")
        import matgl
        from matgl.ext.ase import Relaxer

        _RELAXER = Relaxer(potential=matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2"))
    return _RELAXER


def relax(structure, steps: int = 60):
    """Relax geometry with the MLIP; return (relaxed_structure, max_displacement_A)."""
    import numpy as np

    res = _relaxer().relax(structure, fmax=0.1, steps=steps, verbose=False)
    final = res["final_structure"]
    disp = float(np.max(np.linalg.norm(final.cart_coords - structure.cart_coords, axis=1)))
    return final, disp


def formation_energy_per_atom(structure) -> float:
    """Formation energy (eV/atom) in the Materials Project reference (M3GNet-Eform)."""
    from .screen import _models

    return float(_models()["eform"].predict_structure(structure))


def energy_above_hull(structure, ef_per_atom: float, mpr) -> float:
    """E_hull (eV/atom): candidate formation energy vs the MP convex hull.

    Consistent references: the candidate's formation energy (MP-referenced M3GNet-Eform)
    is converted to a total energy using MP's own elemental reference energies, then
    compared on the same phase diagram as the MP entries.
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    comp = structure.composition
    amt = comp.get_el_amt_dict()
    entries = mpr.get_entries_in_chemsys(sorted(amt))
    pd_ = PhaseDiagram(entries)

    # Build elemental references from the raw entries. PhaseDiagram.el_refs can silently
    # drop elements (e.g. Yb) during GGA/GGA+U mixing, which made el_refs[Element('Yb')]
    # raise KeyError; reading the lowest-energy elemental entry directly is robust to that.
    el_ref: dict[str, float] = {}
    for e in entries:
        if e.composition.is_element:
            sym = e.composition.elements[0].symbol
            el_ref[sym] = min(e.energy_per_atom, el_ref.get(sym, e.energy_per_atom))
    missing = [s for s in amt if s not in el_ref]
    if missing:
        raise ValueError(f"no elemental reference in MP for {missing}")

    e_total = ef_per_atom * comp.num_atoms + sum(amt[s] * el_ref[s] for s in amt)
    # allow_negative so below-hull (very stable) candidates return a value instead of raising.
    _, ehull = pd_.get_decomp_and_e_above_hull(ComputedEntry(comp, e_total), allow_negative=True)
    return float(ehull)


def is_novel_composition(structure, mpr) -> bool:
    """True if no Materials Project entry shares the reduced formula."""
    docs = mpr.materials.summary.search(
        formula=structure.composition.reduced_formula, fields=["material_id"]
    )
    return len(docs) == 0


def _unique_flags(structures) -> list[bool]:
    """Mark structural duplicates among the generated set (keep first occurrence)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    sm = StructureMatcher()
    seen, flags = [], []
    for s in structures:
        dup = any(sm.fit(s, kept) for kept in seen)
        flags.append(not dup)
        if not dup:
            seen.append(s)
    return flags


def evaluate_sun(structures, do_relax: bool = False, relax_steps: int = 60) -> pd.DataFrame:
    """Run S.U.N. on (id, Structure) pairs -> per-candidate DataFrame.

    By default uses the generated structures as-is (MatterGen outputs are already
    near-equilibrium) for speed; pass do_relax=True for an MLIP relaxation first.
    """
    from tqdm import tqdm

    from ..materials.sources import mp_client

    # 1. formation energy (optionally after MLIP relaxation)
    prepared = []
    for sid, s in tqdm(structures, desc=("relax+eform" if do_relax else "eform"), unit="mat"):
        struct, disp = relax(s, steps=relax_steps) if do_relax else (s, 0.0)
        prepared.append((sid, struct, formation_energy_per_atom(struct), disp))

    # 2. uniqueness among the generated set
    unique = _unique_flags([p[1] for p in prepared])

    # 3. stability (E_hull) + novelty vs Materials Project
    rows = []
    with mp_client() as mpr:
        for (sid, struct, ef, disp), uniq in zip(tqdm(prepared, desc="hull+novelty", unit="mat"), unique):
            try:
                ehull = energy_above_hull(struct, ef, mpr)
            except Exception as exc:  # noqa: BLE001 - keep going if one hull/query fails
                ehull = float("nan")
                tqdm.write(f"  {sid}: E_hull failed ({exc})")
            novel = is_novel_composition(struct, mpr)
            stable = bool(ehull < EHULL_STABLE)
            rows.append(
                {
                    "id": sid,
                    "formula": struct.composition.reduced_formula,
                    "e_hull_ev_atom": round(ehull, 3),
                    "relax_disp_A": round(disp, 3),
                    "stable": stable,
                    "unique": bool(uniq),
                    "novel": bool(novel),
                    "SUN": bool(stable and uniq and novel),
                }
            )
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, path=REPORT):
    n = len(df)
    s, u, nov, sun = int(df.stable.sum()), int(df.unique.sum()), int(df.novel.sum()), int(df.SUN.sum())
    lines = [
        "# S.U.N. validation — generated materials",
        "",
        f"Evaluated **{n}** generated structures (M3GNet formation energy + MP convex hull).",
        "",
        f"- **Stable** (E_hull < {EHULL_STABLE} eV/atom): {s}/{n}",
        f"- **Unique** (no duplicate among generated): {u}/{n}",
        f"- **Novel** (composition not in Materials Project): {nov}/{n}",
        f"- **S.U.N. (all three): {sun}/{n}**  ← the headline generative-quality number",
        "",
        "| id | formula | E_hull (eV/atom) | stable | unique | novel | SUN |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.sort_values("SUN", ascending=False).iterrows():
        lines.append(
            f"| {r['id']} | {r['formula']} | {r['e_hull_ev_atom']} | "
            f"{'✓' if r['stable'] else '·'} | {'✓' if r['unique'] else '·'} | "
            f"{'✓' if r['novel'] else '·'} | {'✓' if r['SUN'] else '·'} |"
        )
    lines += [
        "",
        "_E_hull is a fast-screening approximation (MLIP energy vs DFT hull); consistent-reference "
        "DFT would refine the absolute value. Stability + uniqueness + novelty together are the "
        "standard measure of generative-model quality._",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(cif_dir: str | None = None, do_relax: bool = False, relax_steps: int = 60) -> None:
    structures = load_cifs(cif_dir) if cif_dir else demo_structures()
    mode = "MLIP-relax + hull + match" if do_relax else "fast (no relax) + hull + match"
    print(f"S.U.N. on {len(structures)} structures [{mode}]...")
    df = evaluate_sun(structures, do_relax=do_relax, relax_steps=relax_steps)
    print(df.to_string(index=False))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    out = write_report(df)
    print(f"\nS.U.N.: {int(df.SUN.sum())}/{len(df)} stable+unique+novel -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--relax", action="store_true", help="MLIP-relax first (slower, more accurate)")
    ap.add_argument("--relax-steps", type=int, default=60)
    args = ap.parse_args()
    main(cif_dir=None if args.demo else args.cif_dir, do_relax=args.relax, relax_steps=args.relax_steps)
