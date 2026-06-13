"""Load and validate the frozen target spec (configs/targets.yaml).

Every optimizer objective and pass/fail gate reads thresholds from here, so the
spec lives in exactly one place. ``load_targets`` resolves the path relative to
the repo root by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = REPO_ROOT / "configs" / "targets.yaml"
ENV_PATH = REPO_ROOT / ".env"

# Optional dependency: if python-dotenv is installed we auto-load .env; otherwise
# we fall back to the real process environment so the module always imports cleanly.
try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - dotenv is in requirements but keep import-safe
    _load_dotenv = None

_ENV_LOADED = False


def load_env(path: str | Path = ENV_PATH) -> None:
    """Load secrets from .env into the process environment (idempotent).

    Existing environment variables win over .env values, so a key exported in the
    shell or set by the cluster scheduler is never clobbered.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if _load_dotenv is not None and Path(path).exists():
        _load_dotenv(path, override=False)
    _ENV_LOADED = True


def get_secret(name: str, required: bool = True) -> str | None:
    """Return a secret from the environment (loading .env first).

    Raises a clear, actionable error when a required secret is absent instead of
    letting a downstream API call fail with an opaque auth error.
    """
    load_env()
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(
            f"Missing required secret '{name}'. Add it to {ENV_PATH} "
            f"(see .env.example) or export it in your shell."
        )
    return value


def get_mp_api_key() -> str:
    """Materials Project API key (free at https://materialsproject.org)."""
    return get_secret("MP_API_KEY")  # type: ignore[return-value]

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
