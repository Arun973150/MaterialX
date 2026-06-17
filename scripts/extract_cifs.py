"""Turn a MatterGen output dir into a `cifs/` dir of individual *.cif files.

`mattergen-generate <out>` writes the batch as `generated_crystals_cif.zip` (a zip of
per-structure CIFs) and/or `generated_crystals.extxyz`. Our screening reads a directory
of `*.cif` (one per structure), so this normalizes either output into `<out>/cifs/`.

    python scripts/extract_cifs.py /workspace/runs/radar          # -> /workspace/runs/radar/cifs/*.cif
    python scripts/extract_cifs.py /workspace/runs/diel --prefix diel   # names cifs diel_<n>.cif

Uses only the standard library for the zip path (runs in any env). The .extxyz fallback
needs ase + pymatgen (present in the MatterGen / base envs).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def extract(run_dir: str, prefix: str | None = None) -> int:
    run = Path(run_dir)
    if not run.is_dir():
        raise SystemExit(f"not a directory: {run}")
    cifs = run / "cifs"
    cifs.mkdir(exist_ok=True)

    # 1) preferred: the CIF zip MatterGen writes
    zips = sorted(run.glob("*cif*.zip")) + sorted(run.glob("generated_crystals*.zip"))
    if zips:
        n = 0
        with zipfile.ZipFile(zips[0]) as z:
            for m in z.namelist():
                if not m.lower().endswith(".cif"):
                    continue
                name = Path(m).name
                if prefix:
                    name = f"{prefix}_{name}"
                (cifs / name).write_bytes(z.read(m))
                n += 1
        print(f"extracted {n} cif(s) from {zips[0].name} -> {cifs}")
        return n

    # 2) fallback: an extxyz of structures
    ex = sorted(run.glob("*.extxyz"))
    if ex:
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = read(str(ex[0]), index=":")
        for i, a in enumerate(atoms):
            stem = f"{prefix}_gen_{i}" if prefix else f"gen_{i}"
            AseAtomsAdaptor.get_structure(a).to(filename=str(cifs / f"{stem}.cif"))
        print(f"converted {len(atoms)} structure(s) from {ex[0].name} -> {cifs}")
        return len(atoms)

    raise SystemExit(
        f"no MatterGen output found in {run} (looked for *cif*.zip and *.extxyz). "
        f"Contents: {[p.name for p in run.iterdir()]}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="MatterGen output dir (contains the cif zip / extxyz)")
    ap.add_argument("--prefix", default=None, help="prefix cif names (avoids gen_0 collisions when pooling roles)")
    args = ap.parse_args()
    extract(args.run_dir, args.prefix)
