"""Normalize a MatterGen output dir into a `cifs/` dir of individual *.cif files.

`mattergen-generate <out>` writes `generated_crystals_cif.zip` and/or `generated_crystals.extxyz`;
screening reads a directory of `*.cif`. This converts either output into `<out>/cifs/`.
The standard-library zip path runs in any env; the .extxyz fallback needs ase + pymatgen.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def extract(run_dir: str, prefix: str | None = None) -> int:
    run = Path(run_dir)
    if not run.is_dir():
        raise SystemExit(f"not a directory: {run}")
    cifs = run / "cifs"
    cifs.mkdir(exist_ok=True)

    zips = sorted(run.glob("*cif*.zip")) + sorted(run.glob("generated_crystals*.zip"))
    if zips:
        n = 0
        with zipfile.ZipFile(zips[0]) as z:
            for m in z.namelist():
                if not m.lower().endswith(".cif"):
                    continue
                name = f"{prefix}_{Path(m).name}" if prefix else Path(m).name
                (cifs / name).write_bytes(z.read(m))
                n += 1
        print(f"extracted {n} cif(s) from {zips[0].name} -> {cifs}")
        return n

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
