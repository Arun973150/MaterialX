"""#4 multi-band stealth signature objective — minimize reflection + thermal emission.

The gist's objective is "minimize the reflection and thermal-emission signatures of the
generated materials." We do exactly that, on the material, via real predicted properties (#3)
and validated physics:

  * radar:  metal-backed single-layer (Dallenbach) reflection loss from the material's complex
            permittivity eps and permeability mu over X-band (transmission-line formula).
            A deep reflection minimum = good radar absorption.
  * IR:     thermal-IR surface emissivity from the complex dielectric (Fresnel). Low emissivity
            = good IR concealment.

eps'' (dielectric loss) comes from the predicted material class x the measured-literature loss
tangent (Tier A2); mu from the magnetic-class prior. The scalar `stealth_objective` (to MINIMIZE)
blends the bands and can be penalized by weight / (1 - durability) / cost for the RL stage (#5).

Pure-physics helpers (dallenbach_reflection, surface_emissivity) are import-light and tested;
`material_signature` / `stealth_objective` additionally call the #3 predictors.
"""

from __future__ import annotations

import numpy as np

from ..physics.radar import Z0
from .em_literature import LITERATURE, class_mu_prior

C0 = 299_792_458.0
X_BAND_GHZ = np.linspace(8.0, 12.0, 21)


def _class_loss_tangent(mat_class: str) -> float:
    """Median eps''/eps' for a material class, from the measured-literature table (Tier A2)."""
    rows = [r for r in LITERATURE if r.mat_class == mat_class]
    return float(np.median([r.loss_tan_e for r in rows])) if rows else 0.1


def dallenbach_reflection(eps: complex, mu: complex, thickness_mm: float,
                          f_ghz: np.ndarray = X_BAND_GHZ) -> np.ndarray:
    """Reflection loss (dB) of a metal-backed single material layer (transmission-line).

    Z_in = Z0 sqrt(mu/eps) tanh(j (2 pi f d / c) sqrt(mu eps));  RL = 20 log10|(Z_in-Z0)/(Z_in+Z0)|.
    eps, mu are complex relative values; d in mm. This is the standard RAM reflection-loss model.
    """
    f = np.atleast_1d(np.asarray(f_ghz, dtype=float)) * 1e9
    d = thickness_mm * 1e-3
    sqrt_em = np.sqrt(eps * mu)
    z_in = Z0 * np.sqrt(mu / eps) * np.tanh(1j * (2 * np.pi * f * d / C0) * sqrt_em)
    gamma = (z_in - Z0) / (z_in + Z0)
    return 20.0 * np.log10(np.clip(np.abs(gamma), 1e-6, None))


def surface_emissivity(eps: complex) -> float:
    """Normal-incidence thermal-IR emissivity of a material surface from its complex dielectric.

    n_c = sqrt(eps); R = |(n_c - 1)/(n_c + 1)|^2; emissivity = 1 - R. High-loss/metallic surfaces
    reflect (low emissivity = good IR concealment); dielectrics emit more.
    """
    n_c = np.sqrt(complex(eps))
    r = abs((n_c - 1.0) / (n_c + 1.0)) ** 2
    return float(np.clip(1.0 - r, 0.0, 1.0))


def _material_class(props: dict) -> str:
    if props.get("magnetic"):
        return "magnetic"
    return "conductive" if props.get("sigma_class") in ("conductive", "semiconducting") else "dielectric"


def material_signature(structure, thickness_mm: float = 2.0) -> dict:
    """Predict properties (#3) -> compute the material's radar + IR stealth signature."""
    from .predictors import predict_properties

    p = predict_properties(structure)
    mat_class = _material_class(p)
    eps_r = float(p["dielectric_eps"])
    eps = complex(eps_r, eps_r * _class_loss_tangent(mat_class))
    mu = complex(p["mu_real"], p["mu_imag"]) if p["magnetic"] else complex(1.0, 0.0)

    rl = dallenbach_reflection(eps, mu, thickness_mm)
    radar_min_rl_db = float(rl.min())
    radar_badness = float(10 ** (radar_min_rl_db / 20.0))   # |Gamma| at best absorption (0 good, 1 bad)
    ir_emissivity = surface_emissivity(eps)

    return {
        "material_class": mat_class,
        "eps": eps, "mu": mu,
        "radar_min_rl_db": round(radar_min_rl_db, 2),
        "radar_badness": round(radar_badness, 4),
        "ir_emissivity": round(ir_emissivity, 4),
        "durability_score": p["durability_score"],
        "properties": p,
    }


# per-role weighting: which band the material's role is judged on (radar vs IR), 0..1.
_ROLE_WEIGHTS = {
    "radar_conductor": (0.85, 0.15),
    "radar_magnetic": (0.85, 0.15),
    "ir_phasechange": (0.15, 0.85),
    "dielectric_spacer": (0.5, 0.5),
}


def stealth_objective(structure, role: str | None = None, thickness_mm: float = 2.0,
                      weight_penalty: float = 0.0, durability_penalty: float = 0.1) -> dict:
    """Scalar objective to MINIMIZE (lower = stealthier), role-weighted across bands.

    objective = w_radar*radar_badness + w_ir*ir_emissivity + durability_penalty*(1 - durability)
    (+ weight_penalty hook for #5). Returns the scalar plus the full signature breakdown.
    """
    sig = material_signature(structure, thickness_mm)
    w_r, w_ir = _ROLE_WEIGHTS.get(role, (0.5, 0.5))
    obj = (w_r * sig["radar_badness"] + w_ir * sig["ir_emissivity"]
           + durability_penalty * (1.0 - sig["durability_score"]))
    sig["objective"] = round(float(obj), 4)
    sig["role_weights"] = {"radar": w_r, "ir": w_ir}
    return sig
