"""#10 experimental anchor — validate the radar physics against MEASURED absorber data.

Everything else is model-vs-model (ECM vs openEMS). This closes the loop to reality: feed the
*measured* complex permittivity/permeability of a real radar absorber (Tier A2,
`em_literature.py`) through the metal-backed single-layer (Dallenbach) reflection model and check
it reproduces the absorber's *known* behavior — a deep reflection minimum at the impedance-/
quarter-wave-matched thickness in X-band, consistent with the published literature.

This is a literature anchor (measured material constants -> reproduced reflection), not a lab
measurement; lab synthesis + measurement remains the future closed-loop step.

    python -m stealth.discovery.anchor            # run the anchor on carbonyl iron + ferrites
"""

from __future__ import annotations

import numpy as np

from .em_literature import LITERATURE, reference_absorber
from .objective import dallenbach_reflection

F_GHZ = np.linspace(2.0, 18.0, 161)


def optimal_thickness(eps: complex, mu: complex, f_ghz=F_GHZ,
                      t_range_mm=(0.3, 6.0), n=120) -> dict:
    """Scan layer thickness; return the thickness giving the deepest reflection minimum."""
    ts = np.linspace(*t_range_mm, n)
    best = None
    for t in ts:
        rl = dallenbach_reflection(eps, mu, t, f_ghz)
        i = int(np.argmin(rl))
        cand = {"thickness_mm": float(t), "min_rl_db": float(rl[i]), "f_at_min_ghz": float(f_ghz[i])}
        if best is None or cand["min_rl_db"] < best["min_rl_db"]:
            best = cand
    return best


def quarter_wave_thickness(eps: complex, mu: complex, f_ghz: float) -> float:
    """Theory matching thickness d = c / (4 f Re(sqrt(eps*mu))) in mm (lambda/4 in the medium)."""
    n_eff = float(np.real(np.sqrt(eps * mu)))
    return 299_792_458.0 / (4.0 * f_ghz * 1e9 * n_eff) * 1e3


def reproduce(record, f_ghz=F_GHZ) -> dict:
    """Reproduce a measured absorber's reflection from its eps,mu; compare optimum vs lambda/4 theory."""
    eps = complex(record.eps_real, -abs(record.eps_imag))   # e^{+jwt} convention: lossy -> negative imag
    mu = complex(record.mu_real, -abs(record.mu_imag))
    best = optimal_thickness(eps, mu, f_ghz)
    qw = quarter_wave_thickness(eps, mu, best["f_at_min_ghz"])
    return {
        "material": record.material, "class": record.mat_class,
        "eps": eps, "mu": mu,
        "best_thickness_mm": round(best["thickness_mm"], 2),
        "min_rl_db": round(best["min_rl_db"], 1),
        "f_at_min_ghz": round(best["f_at_min_ghz"], 1),
        "quarter_wave_thickness_mm": round(qw, 2),
    }


def validate(min_absorption_db: float = -15.0) -> dict:
    """Anchor check: a real magnetic absorber must reproduce strong, well-matched absorption."""
    ref = reference_absorber()                 # carbonyl iron (measured eps,mu)
    r = reproduce(ref)
    # the scanned optimum thickness should be physically sensible and near the lambda/4 estimate
    ratio = r["best_thickness_mm"] / max(r["quarter_wave_thickness_mm"], 1e-6)
    r["passes"] = bool(r["min_rl_db"] <= min_absorption_db and 0.5 <= ratio <= 2.0)
    r["thickness_vs_quarterwave_ratio"] = round(ratio, 2)
    return r


def main() -> None:
    print("#10 experimental anchor — measured eps,mu -> reproduced reflection\n")
    print(f"{'material':40} {'class':10} {'d_opt':>6} {'minRL':>7} {'f_min':>6} {'d_lambda/4':>10}")
    for rec in LITERATURE:
        if rec.mat_class == "magnetic" or rec.eps_imag > 3:   # absorbers worth a 1-layer match
            r = reproduce(rec)
            print(f"{r['material']:40} {r['class']:10} {r['best_thickness_mm']:6.2f} "
                  f"{r['min_rl_db']:7.1f} {r['f_at_min_ghz']:6.1f} {r['quarter_wave_thickness_mm']:10.2f}")
    v = validate()
    print(f"\nAnchor ({v['material']}): min RL {v['min_rl_db']} dB at {v['best_thickness_mm']} mm "
          f"(lambda/4 theory {v['quarter_wave_thickness_mm']} mm, ratio {v['thickness_vs_quarterwave_ratio']}) "
          f"-> {'PASS' if v['passes'] else 'CHECK'}")


if __name__ == "__main__":
    main()
