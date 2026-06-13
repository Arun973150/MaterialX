"""Material registry: the catalog of candidate materials per stack layer.

Loads ``configs/materials.yaml`` into validated :class:`Material` records. Each
material is either backed by refractiveindex.info (wavelength-resolved n,k) or by
documented literature values (for materials not in that database).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import REPO_ROOT

DEFAULT_MATERIALS = REPO_ROOT / "configs" / "materials.yaml"

LAYER_ROLES = {
    "visible_ec",
    "ir_thermochromic",
    "dielectric_spacer",
    "radar_conductor",
    "substrate",
    "ground",
}
SOURCES = {"refractiveindex", "literature"}


@dataclass(frozen=True)
class Material:
    """A candidate material (or one phase/state of one) for a stack layer."""

    name: str
    layer_role: str
    source: str
    provenance: str
    ri: dict[str, str] | None = None          # {shelf, book, page} for refractiveindex
    state: str | None = None                  # e.g. insulating / metallic for VO2
    literature_nk: dict[str, float] | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer_role not in LAYER_ROLES:
            raise ValueError(f"{self.name}: unknown layer_role {self.layer_role!r}")
        if self.source not in SOURCES:
            raise ValueError(f"{self.name}: unknown source {self.source!r}")
        if self.source == "refractiveindex" and not self.ri:
            raise ValueError(f"{self.name}: source=refractiveindex requires an 'ri' key")
        if self.source == "refractiveindex":
            missing = {"shelf", "book", "page"} - set(self.ri or {})
            if missing:
                raise ValueError(f"{self.name}: ri missing {missing}")


def load_registry(path: str | Path = DEFAULT_MATERIALS) -> list[Material]:
    """Read and validate the material catalog."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return [Material(**entry) for entry in raw["materials"]]


def by_role(materials: list[Material], role: str) -> list[Material]:
    """All materials serving a given layer role."""
    if role not in LAYER_ROLES:
        raise ValueError(f"unknown layer_role {role!r}")
    return [m for m in materials if m.layer_role == role]


if __name__ == "__main__":
    mats = load_registry()
    print(f"{len(mats)} materials:")
    for role in sorted(LAYER_ROLES):
        names = [m.name for m in by_role(mats, role)]
        print(f"  {role:18s}: {', '.join(names) or '-'}")
