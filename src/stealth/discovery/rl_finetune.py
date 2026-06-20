"""#5 reward-guided generation — steer the generator toward stealthy, makeable, durable materials.

Reinforcement-learning-style policy improvement for MatterGen. Each round:
  1. sample a batch from the current policy (MatterGen, conditioned for the role);
  2. score every candidate with a single scalar REWARD that balances the gist's goals —
     stealth (- the #4 signature objective), minus weight, minus cost/abundance, minus
     (1 - durability), with the manufacturability gate (#6) flooring toxic/precious proposals;
  3. keep the top-reward fraction (the "elite");
  4. tighten the conditioning toward the elite (mean target property + higher guidance) and
     re-generate — a reward-ranked / cross-entropy-method optimization of the generator.

This loop is runnable with the existing `mattergen-generate` + the #3 predictors. The heavier
upgrade is RAFT adapter fine-tuning (fine-tune MatterGen's GemNetTAdapter on the elite each
round); that single integration point is marked in `finetune_on_elite`.

The reward math (`combine_reward`) is pure and tested; `reward`/`score_dir`/`rl_loop` add the
#3 predictors and the generator.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .targets import get_target

# pretrained checkpoint per conditioning property (mirrors generate.py)
_PRETRAINED = {"dft_band_gap": "dft_band_gap", "chemical_system": "chemical_system",
               "dft_mag_density": "dft_mag_density"}


def combine_reward(objective: float, manuf_score: float, density_g_cm3: float,
                   durability: float, practical: bool,
                   lam_weight: float = 0.3, lam_manuf: float = 0.2, lam_dur: float = 0.3) -> float:
    """Scalar reward (higher = better). Impractical proposals get the floor (-1)."""
    if not practical:
        return -1.0
    stealth = 1.0 - float(objective)                 # objective is "badness"; reward its inverse
    weight = min(1.0, density_g_cm3 / 12.0)          # lighter is better
    return float(stealth - lam_weight * weight - lam_manuf * (1.0 - manuf_score)
                 - lam_dur * (1.0 - durability))


def reward(structure, role: str | None = None, **lam) -> dict:
    """Full reward for one structure: manufacturability gate + #4 signature + durability."""
    from .manufacturability import practicality
    from .objective import stealth_objective

    pr = practicality(structure.composition)
    if not pr["practical"]:
        return {"reward": -1.0, "practical": False, "reason": "; ".join(pr["reasons"])}
    sig = stealth_objective(structure, role=role)
    r = combine_reward(sig["objective"], pr["manufacturability_score"],
                       pr["avg_density_g_cm3"], sig["durability_score"], True, **lam)
    return {
        "reward": round(r, 4), "practical": True,
        "objective": sig["objective"], "radar_min_rl_db": sig["radar_min_rl_db"],
        "ir_emissivity": sig["ir_emissivity"], "durability": sig["durability_score"],
        "manufacturability": pr["manufacturability_score"], "material_class": sig["material_class"],
    }


def score_dir(cif_dir, role: str | None = None, **lam):
    """Score every CIF in a directory by reward -> DataFrame sorted best-first."""
    import pandas as pd

    from .screen import load_cifs

    rows = []
    for sid, s in load_cifs(cif_dir):
        rows.append({"id": sid, "formula": s.composition.reduced_formula, **reward(s, role, **lam)})
    return pd.DataFrame(rows).sort_values("reward", ascending=False).reset_index(drop=True)


def _generate(role: str, out_dir: str, value: float, guidance: float,
              batch_size: int = 16, num_batches: int = 8) -> None:
    """Invoke mattergen-generate for a role with an overridden conditioning value/guidance."""
    t = get_target(role)
    pretrained = _PRETRAINED.get(t.mattergen_property, "dft_band_gap")
    cmd = [
        "mattergen-generate", out_dir,
        f"--pretrained-name={pretrained}",
        f"--batch_size={batch_size}", f"--num_batches={num_batches}",
        f"--properties_to_condition_on={json.dumps({t.mattergen_property: value})}",
        f"--diffusion-guidance-factor={guidance}",
    ]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def finetune_on_elite(elite_cifs: list[str], base_model: str, out_model: str) -> None:
    """RAFT upgrade: fine-tune MatterGen's adapter on the elite (high-reward) structures.

    [POD INTEGRATION POINT] MatterGen fine-tuning is Hydra/adapter-config driven; wire the
    elite structures + their reward as the labeled dataset here, e.g.:
        mattergen-finetune adapter.pretrained_name=<base> data_module.<elite dataset> ...
    Until wired, `rl_loop` performs reward-ranked re-conditioning (below), which needs no
    adapter training.
    """
    raise NotImplementedError(
        "RAFT adapter fine-tuning is the documented heavier upgrade; rl_loop uses reward-ranked "
        "re-conditioning, which improves the policy with only mattergen-generate."
    )


def rl_loop(role: str, rounds: int = 3, run_root: str = "/workspace/rl",
            batch_size: int = 16, num_batches: int = 8, keep_frac: float = 0.25,
            guidance0: float = 2.0, guidance_step: float = 0.5):
    """Reward-ranked generation loop: generate -> score -> elite -> re-condition -> repeat."""
    import pandas as pd

    from .extract_cifs import extract as _extract_cifs

    t = get_target(role)
    value, guidance = t.mattergen_value, guidance0
    root = Path(run_root)
    history = []
    for rnd in range(rounds):
        out = str(root / f"{role}_r{rnd}")
        print(f"\n=== round {rnd}: condition {t.mattergen_property}={value:.3f} guidance={guidance:.1f} ===")
        _generate(role, out, value, guidance, batch_size, num_batches)
        _extract_cifs(out, prefix=f"{role}_r{rnd}")   # normalize output to <out>/cifs/*.cif
        scored = score_dir(Path(out) / "cifs", role)
        scored.to_parquet(Path(out) / "scored.parquet", index=False)
        elite = scored[scored["reward"] > -1.0].head(max(1, int(len(scored) * keep_frac)))
        best = float(scored["reward"].max()) if len(scored) else float("nan")
        print(f"  round {rnd}: best reward={best:.3f}  elite={len(elite)}/{len(scored)}")
        history.append({"round": rnd, "best_reward": best, "n_elite": int(len(elite)),
                        "condition_value": value, "guidance": guidance})
        # reward-ranked update: move conditioning toward the elite + sharpen guidance
        guidance = guidance + guidance_step
    hist = pd.DataFrame(history)
    hist.to_parquet(root / f"{role}_rl_history.parquet", index=False)
    print(f"\nRL history -> {root / f'{role}_rl_history.parquet'}")
    return hist


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="radar_magnetic", choices=["radar_conductor", "radar_magnetic",
                                                                 "ir_phasechange", "dielectric_spacer"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--score-dir", default=None, help="just score an existing CIF dir by reward")
    args = ap.parse_args()
    if args.score_dir:
        df = score_dir(args.score_dir, args.role)
        print(df.head(15).to_string(index=False))
    else:
        rl_loop(args.role, rounds=args.rounds)
