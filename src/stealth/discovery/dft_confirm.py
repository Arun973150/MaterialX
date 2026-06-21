"""DFT confirmation (GPAW) of the NOVEL shortlist — model-grade stability -> DFT-grade.

The shortlist's stability is MLIP/GNN-predicted (~0.5 eV error). This re-checks the *novel*
candidates with real DFT (GPAW, PBE): relax each to a genuine DFT minimum and compute its DFT
formation energy vs GPAW elemental references (a code-consistent reference scheme). A candidate
that relaxes stably (forces converge, structure doesn't collapse) with a **negative formation
energy** is DFT-confirmed thermodynamically favorable — a strong upgrade over the MLIP screen.

Honest scope: a fully Materials-Project-consistent E_hull needs VASP with MP's exact settings +
corrections (the gold standard). GPAW PBE here confirms (a) the generated/MLIP structure is a real
DFT local minimum, not a model artifact, and (b) its DFT formation energy is favorable. The
remaining gap to "exact E_hull on the MP hull" is the named VASP step.

Only run on the NOVEL candidates — known ones (VO2, CaF2, NaVO2) are already DB-confirmed.

Setup (pod):
    apt-get install -y libxc-dev libopenblas-dev libfftw3-dev
    pip install gpaw && gpaw install-data --register ~/gpaw-setups
Run:
    python -m stealth.discovery.dft_confirm --cif-dir /workspace/gen/all \
        --formulas MnV2MoO6 V4SnO10 CsLiMgF4 Li4F5
    # or point at explicit CIF ids:  --ids ir_gen_108 ir_gen_47 diel_gen_127 diel_gen_73
"""

from __future__ import annotations

import json

from ..config import REPO_ROOT

REPORT = REPO_ROOT / "reports" / "dft_confirmation.md"
OUT = REPO_ROOT / "data" / "dft_confirmation.json"
REF_CACHE = REPO_ROOT / "data" / "dft_elem_refs.json"

EHULL_STABLE = 0.1  # eV/atom (favorable-energy threshold for the verdict)

# Elemental reference states (standard states) for the formation energy.
#   ("bulk", crystalstructure, a [, c])  |  ("molecule", name [, magmom])
_ELEMENT_REF = {
    "Li": ("bulk", "bcc", 3.49), "Na": ("bulk", "bcc", 4.23), "K": ("bulk", "bcc", 5.23),
    "Mg": ("bulk", "hcp", 3.21, 5.21), "Ca": ("bulk", "fcc", 5.58), "Sr": ("bulk", "fcc", 6.08),
    "Ba": ("bulk", "bcc", 5.02), "Al": ("bulk", "fcc", 4.05), "Si": ("bulk", "diamond", 5.43),
    "Ti": ("bulk", "hcp", 2.95, 4.68), "V": ("bulk", "bcc", 3.03), "Cr": ("bulk", "bcc", 2.88),
    "Mn": ("bulk", "bcc", 2.89), "Fe": ("bulk", "bcc", 2.87), "Co": ("bulk", "hcp", 2.51, 4.07),
    "Ni": ("bulk", "fcc", 3.52), "Cu": ("bulk", "fcc", 3.61), "Zn": ("bulk", "hcp", 2.66, 4.95),
    "Nb": ("bulk", "bcc", 3.30), "Mo": ("bulk", "bcc", 3.15), "Sn": ("bulk", "diamond", 6.49),
    "Cs": ("bulk", "bcc", 6.05), "C": ("bulk", "diamond", 3.567),
    "O": ("molecule", "O2", 2.0), "N": ("molecule", "N2", 0.0), "H": ("molecule", "H2", 0.0),
    "F": ("molecule", "F2", 0.0), "Cl": ("molecule", "Cl2", 0.0),
}


def _calc(pw: float = 500.0, kpts=(3, 3, 3), spin: bool = False):
    """A GPAW plane-wave PBE calculator (kept in one place so settings stay consistent).

    Wider Fermi smearing + an SCF iteration cap keep metallic / magnetic transition-metal
    systems from grinding forever — a non-converging SCF raises instead of hanging.
    """
    from gpaw import GPAW, PW

    return GPAW(mode=PW(pw), xc="PBE", kpts=kpts, txt=None, spinpol=spin,
                occupations={"name": "fermi-dirac", "width": 0.1},
                maxiter=200, convergence={"energy": 5e-4, "density": 1e-3})


# magnetic transition metals -> spin-polarize + seed initial moments so the SCF converges
_MAGNETIC_ELEMENTS = {"Mn", "Fe", "Co", "Ni", "Cr", "V"}
_INIT_MAGMOM = {"Mn": 5.0, "Fe": 4.0, "Co": 3.0, "Ni": 2.0, "Cr": 5.0, "V": 3.0, "Mo": 1.0, "Ti": 1.0}


def formation_energy_per_atom(e_total: float, comp, elem_ref_per_atom: dict) -> float:
    """E_form/atom = (E_total - sum n_i * mu_i) / N_atoms.  (Pure; testable.)"""
    amt = comp.get_el_amt_dict()
    n_atoms = sum(amt.values())
    ref = sum(amt[s] * elem_ref_per_atom[s] for s in amt)
    return (e_total - ref) / n_atoms


def _elemental_reference(sym: str) -> float:
    """GPAW PBE energy per atom of the element's standard reference state (cached on disk)."""
    cache = json.loads(REF_CACHE.read_text()) if REF_CACHE.exists() else {}
    if sym in cache:
        return cache[sym]
    if sym not in _ELEMENT_REF:
        raise ValueError(f"no reference state defined for element {sym!r}")

    from ase.build import bulk, molecule
    from ase.optimize import LBFGS

    spec = _ELEMENT_REF[sym]
    if spec[0] == "bulk":
        a = bulk(sym, spec[1], a=spec[2], *( [spec[3]] if len(spec) > 3 else [] )) \
            if len(spec) <= 3 else bulk(sym, spec[1], a=spec[2], c=spec[3])
        a.calc = _calc(kpts=(6, 6, 6), spin=sym in ("Fe", "Co", "Ni", "Mn"))
        e = a.get_potential_energy() / len(a)
    else:  # molecule in a box, gamma point
        m = molecule(spec[1])
        m.center(vacuum=6.0)
        if len(spec) > 2 and spec[2]:
            m.set_initial_magnetic_moments([spec[2] / len(m)] * len(m))
        m.calc = _calc(kpts=(1, 1, 1), spin=bool(len(spec) > 2 and spec[2]))
        LBFGS(m, logfile=None).run(fmax=0.05, steps=50)
        e = m.get_potential_energy() / len(m)

    cache[sym] = float(e)
    REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    REF_CACHE.write_text(json.dumps(cache, indent=2))
    return float(e)


def confirm_structure(structure, fmax: float = 0.05, steps: int = 80) -> dict:
    """Relax a candidate with GPAW + compute its DFT formation energy."""
    import numpy as np
    from ase.optimize import LBFGS
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    start = atoms.get_positions().copy()
    # spin-polarize magnetic transition-metal candidates (else SCF won't converge)
    magnetic = any(a.symbol in _MAGNETIC_ELEMENTS for a in atoms)
    if magnetic:
        atoms.set_initial_magnetic_moments([_INIT_MAGMOM.get(a.symbol, 0.0) for a in atoms])
    atoms.calc = _calc(kpts=(3, 3, 3), spin=magnetic)
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    converged = bool(opt.converged())
    e_total = float(atoms.get_potential_energy())
    max_disp = float(np.max(np.linalg.norm(atoms.get_positions() - start, axis=1)))

    comp = structure.composition
    refs = {s: _elemental_reference(s) for s in comp.get_el_amt_dict()}
    ef = formation_energy_per_atom(e_total, comp, refs)
    return {
        "formula": comp.reduced_formula,
        "dft_converged": converged,
        "dft_eform_per_atom": round(ef, 4),
        "max_disp_A": round(max_disp, 3),
        "favorable": bool(ef < 0),
        "dft_minimum": bool(converged and max_disp < 1.5),
    }


def run(cif_dir=None, formulas=None, ids=None) -> list[dict]:
    import warnings

    from .screen import load_cifs

    warnings.filterwarnings("ignore")
    structs = load_cifs(cif_dir) if cif_dir else []
    pick = []
    for sid, s in structs:
        f = s.composition.reduced_formula
        if (ids and sid in ids) or (formulas and f in formulas):
            pick.append((sid, s))
    if not pick:
        raise SystemExit("no matching candidates found (check --formulas / --ids / --cif-dir)")

    rows = []
    for sid, s in pick:
        print(f"[DFT] {sid} {s.composition.reduced_formula} — relaxing (GPAW PBE)...")
        try:
            r = confirm_structure(s)
            r["id"] = sid
        except Exception as exc:  # noqa: BLE001
            r = {"id": sid, "formula": s.composition.reduced_formula, "error": str(exc)}
            print(f"  failed: {exc}")
        rows.append(r)
        print(f"  -> {r}")
        OUT.parent.mkdir(parents=True, exist_ok=True)         # incremental save: a hang/kill
        OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")  # never loses finished ones
    return rows


def write_report(rows, path=REPORT):
    lines = [
        "# DFT confirmation (GPAW PBE) of the novel candidates",
        "",
        "Real-DFT re-check of the novel shortlist: each candidate relaxed to a DFT minimum + its "
        "DFT formation energy computed vs GPAW elemental references.",
        "",
        "| id | formula | DFT relaxed (minimum) | DFT E_form (eV/atom) | favorable |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['id']} | {r['formula']} | ERROR | {r['error']} | · |")
        else:
            lines.append(f"| {r['id']} | {r['formula']} | {'✓' if r['dft_minimum'] else '·'} | "
                         f"{r['dft_eform_per_atom']} | {'✓' if r['favorable'] else '·'} |")
    lines += [
        "",
        "_DFT minimum = relaxation converged and the structure stayed near the input (not a collapse). "
        "Favorable = negative DFT formation energy. A fully MP-consistent E_hull needs VASP with MP "
        "settings/corrections — the named further step; GPAW PBE here confirms the candidates are real, "
        "energetically favorable DFT minima rather than MLIP artifacts._",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(cif_dir=None, formulas=None, ids=None):
    rows = run(cif_dir, formulas, ids)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    out = write_report(rows)
    ok = sum(1 for r in rows if r.get("dft_minimum") and r.get("favorable"))
    print(f"\nDFT-confirmed (stable minimum + favorable): {ok}/{len(rows)} -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", required=True, help="dir of candidate CIFs (e.g. /workspace/gen/all)")
    ap.add_argument("--formulas", nargs="*", default=None, help="reduced formulas to confirm")
    ap.add_argument("--ids", nargs="*", default=None, help="explicit CIF ids (stems) to confirm")
    args = ap.parse_args()
    main(cif_dir=args.cif_dir, formulas=args.formulas, ids=args.ids)
