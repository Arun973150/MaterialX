"""Load and validate the frozen target spec (configs/targets.yaml).

Every optimizer objective and pass/fail gate reads thresholds from here, so the
spec lives in exactly one place. ``load_targets`` resolves the path relative to
the repo root by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = REPO_ROOT / "configs" / "targets.yaml"

# Bands that every candidate stack must satisfy simultaneously.
REQUIRED_BANDS = ("radar", "ir_mwir", "ir_lwir", "visible", "nir", "physical")


def load_targets(path: str | Path | None = None) -> dict[str, Any]:
    """Read targets.yaml into a dict and sanity-check required bands exist."""
    path = Path(path) if path is not None else DEFAULT_TARGETS
    if not path.exists():
        raise FileNotFoundError(f"target spec not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        targets = yaml.safe_load(fh)

    missing = [b for b in REQUIRED_BANDS if b not in targets]
    if missing:
        raise ValueError(f"targets.yaml missing required bands: {missing}")
    return targets


if __name__ == "__main__":
    import json

    print(json.dumps(load_targets(), indent=2))
