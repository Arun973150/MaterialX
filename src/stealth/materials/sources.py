"""Data-source clients for the material shortlist (Phase 1).

Wraps the Materials Project (`mp-api`) behind a context manager that pulls the
API key from the environment/.env, plus a tiny ``verify_mp_key`` that confirms the
key works before we lean on it for real queries.

Run a quick check once your key is in .env:

    python -m stealth.materials.sources
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from ..config import get_mp_api_key


@contextmanager
def mp_client() -> Iterator["object"]:
    """Yield an authenticated Materials Project ``MPRester`` session.

    Raises a clear error if `mp-api` isn't installed or the key is missing,
    rather than a deep stack trace from inside the client.
    """
    try:
        from mp_api.client import MPRester
    except ImportError as exc:  # pragma: no cover - exercised on the cluster
        raise ImportError(
            "mp-api is not installed. Run: pip install mp-api"
        ) from exc

    api_key = get_mp_api_key()
    with MPRester(api_key=api_key) as mpr:
        yield mpr


def verify_mp_key() -> dict:
    """Tiny round-trip query (silicon, mp-149) to confirm the key authenticates.

    Returns the fetched record on success; raises on auth/network failure.
    """
    with mp_client() as mpr:
        docs = mpr.materials.summary.search(
            material_ids=["mp-149"],
            fields=["material_id", "formula_pretty", "band_gap"],
        )
    if not docs:
        raise RuntimeError("MP query returned no results — key may lack access.")
    doc = docs[0]
    return {
        "material_id": str(doc.material_id),
        "formula": doc.formula_pretty,
        "band_gap": doc.band_gap,
    }


if __name__ == "__main__":
    try:
        result = verify_mp_key()
    except Exception as exc:  # noqa: BLE001 - surface any failure plainly to the user
        print(f"[FAIL] Materials Project key check failed: {exc}")
        raise SystemExit(1)
    print(f"[OK] Materials Project key works. Sample fetch: {result}")
