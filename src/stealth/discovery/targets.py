"""Per-layer proxy property targets for generative discovery.

MatterGen conditions on *intrinsic* crystal properties (band gap, magnetic density,
chemical system, ...), not on "radar absorption". So we steer generation toward the
intrinsic proxy that makes a good material for each stack layer, then screen on it:

  radar_conductor   -> near-zero band gap  (metallic / lossy -> high microwave loss)
  ir_phasechange    -> narrow gap + V-O    (VO2-like; switchable IR emissivity)
  dielectric_spacer -> wide band gap       (transparent, low-loss spacer)

The visible electrochromic layer is an *organic polymer* — outside MatterGen's
inorganic-crystal scope — so it is intentionally not a generation target here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayerTarget:
    role: str
    description: str
    band_gap_ev: tuple[float, float]          # acceptable (min, max) for screening
    mattergen_property: str                    # which pretrained conditioning to use
    mattergen_value: float                     # target value for that property
    chemical_system: list[str] = field(default_factory=list)  # optional element constraint
    max_eform_per_atom: float = 0.0            # stability proxy: keep eform <= this
    guidance: float = 2.0                      # MatterGen classifier-free guidance strength


TARGETS: dict[str, LayerTarget] = {
    "radar_conductor": LayerTarget(
        role="radar_conductor",
        description="Conductive/dielectric-loss phase for microwave loss (carbide/MXene/SiC-like).",
        band_gap_ev=(0.0, 0.5),
        mattergen_property="dft_band_gap",
        mattergen_value=0.0,
        # abundant, cheap conductive-loss chemistries (carbides, nitrides, doped oxides)
        chemical_system=["Ti", "V", "Cr", "Fe", "Nb", "Si", "Al", "C", "N"],
        max_eform_per_atom=-0.2,
        guidance=2.0,
    ),
    "radar_magnetic": LayerTarget(
        role="radar_magnetic",
        description="Magnetic-loss absorber (spinel/hex ferrite, carbonyl-iron-like) — the "
                    "dominant low-GHz RAM mechanism via complex permeability.",
        band_gap_ev=(0.0, 3.0),
        mattergen_property="dft_mag_density",   # steer toward high magnetization
        mattergen_value=0.2,
        chemical_system=["Fe", "Co", "Ni", "Mn", "Zn", "Mg", "Ba", "Sr", "O"],
        max_eform_per_atom=-1.0,
        guidance=2.0,
    ),
    "ir_phasechange": LayerTarget(
        role="ir_phasechange",
        description="Narrow-gap transition-metal oxide (VO2-like switchable IR emissivity).",
        band_gap_ev=(0.0, 1.0),
        mattergen_property="dft_band_gap",
        mattergen_value=0.6,
        chemical_system=["V", "Ti", "Nb", "Mn", "Fe", "O"],
        max_eform_per_atom=-1.0,
        guidance=2.0,
    ),
    "dielectric_spacer": LayerTarget(
        role="dielectric_spacer",
        description="Wide-gap, low-loss dielectric for the impedance-matching spacer.",
        band_gap_ev=(3.0, 12.0),
        mattergen_property="dft_band_gap",
        mattergen_value=5.0,
        chemical_system=["Si", "Al", "Mg", "Ca", "O", "N"],
        max_eform_per_atom=-1.5,
        guidance=2.0,
    ),
}


def get_target(role: str) -> LayerTarget:
    if role not in TARGETS:
        raise ValueError(f"unknown discovery role {role!r}; choose from {list(TARGETS)}")
    return TARGETS[role]
