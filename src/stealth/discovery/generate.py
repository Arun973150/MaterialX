"""MatterGen generation driver (runs on the A100, not this Windows box).

Builds the property-conditioned MatterGen generation command for a given layer role,
so the A100 generates novel inorganic crystals steered toward the right intrinsic
property (e.g. near-zero band gap for radar conductors). The generated CIFs are then
screened locally by `screen.py`.

On the A100:
    python -m stealth.discovery.generate --role radar_conductor --n 64 --out runs/radar --run
Locally (just print the exact command to run there):
    python -m stealth.discovery.generate --role radar_conductor --n 64 --out runs/radar
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .targets import TARGETS, get_target

# Pretrained MatterGen checkpoint per conditioning property (Microsoft release names).
_PRETRAINED = {
    "dft_band_gap": "dft_band_gap",
    "chemical_system": "chemical_system",
    "dft_mag_density": "dft_mag_density",
}


def generation_command(role: str, n: int, out_dir: str, batch_size: int = 16) -> list[str]:
    """Construct the `mattergen-generate` CLI for a role's property target."""
    t = get_target(role)
    pretrained = _PRETRAINED.get(t.mattergen_property, "dft_band_gap")
    num_batches = max(1, -(-n // batch_size))  # ceil
    cond = {t.mattergen_property: t.mattergen_value}
    return [
        "mattergen-generate",
        str(out_dir),
        f"--pretrained-name={pretrained}",
        f"--batch_size={batch_size}",
        f"--num_batches={num_batches}",
        f"--properties_to_condition_on={json.dumps(cond)}",
        f"--diffusion-guidance-factor={t.guidance}",
    ]


def main(role: str, n: int, out: str, run: bool) -> None:
    cmd = generation_command(role, n, out)
    t = get_target(role)
    print(f"Role: {role} — {t.description}")
    print(f"Conditioning: {t.mattergen_property} = {t.mattergen_value} (guidance {t.guidance})")
    print("Command:\n  " + " ".join(cmd))

    if not run:
        print("\n[dry-run] Add --run on the A100 to execute. Then screen the output:")
        print(f"  python -m stealth.discovery.screen --role {role} --cif-dir {out}")
        return

    if shutil.which("mattergen-generate") is None:
        raise SystemExit(
            "mattergen not installed. On the A100:\n"
            "  pip install git+https://github.com/microsoft/mattergen.git\n"
            "  (CUDA + downloaded checkpoints required)"
        )
    Path(out).mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"\nDone. Screen with:\n  python -m stealth.discovery.screen --role {role} --cif-dir {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="radar_conductor", choices=list(TARGETS))
    ap.add_argument("--n", type=int, default=64, help="number of structures to generate")
    ap.add_argument("--out", default="runs/generated", help="output dir for CIFs")
    ap.add_argument("--run", action="store_true", help="execute (A100 only); otherwise dry-run")
    args = ap.parse_args()
    main(args.role, args.n, args.out, args.run)
