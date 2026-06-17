# Upgrade Roadmap — getting the discovery pipeline closer to real

Researched best approach for each Tier-1 accuracy upgrade. All tooling is open-source.
Key reprioritization: items 1, 3, 4 are high-value and tractable on the A100; item 2 (radar
material prediction) is deliberately **deferred** — see why below.

---

## 1. Optical n,k GNN  —  ✅ build first (highest value, lowest effort)

**Best approach: GNNOpt (pretrained), optionally fine-tuned on JARVIS.**
- **Tool:** [GNNOpt](https://github.com/nguyen-group/GNNOpt) — equivariant GNN, *Advanced Materials* 2024
  ([arXiv 2406.16654](https://arxiv.org/abs/2406.16654)). Predicts the **full optical response**
  (complex dielectric function, complex refractive index n+ik, absorption, reflectance) directly
  from crystal structure, and is **Kramers–Krönig consistent** by construction.
- **Why it's the one:** pretrained models are published (n, k, reflectance), trained on 944 MP
  dielectric functions — so we *use it*, not train from scratch. Directly replaces the first-order
  `optical_bridge.estimate_nk` (Moss/Drude/Tauc).
- **Higher-data alternative:** ALIGNN trained on ~7,000 JARVIS-DFT dielectric functions
  ([arXiv 2510.08738](https://arxiv.org/abs/2510.08738)) — more data, but we'd train it.
- **Effort:** LOW. Run GNNOpt inference on generated CIFs → n,k → existing TMM physics.
- **Integration:** isolated env (e3nn deps), file hand-off like MatterGen.

## 2. Radar EM-property prediction  —  ⏸ deferred (lowest value here)

**Best approach: don't predict it per-material yet — keep sheet resistance as a design knob.**
- **Why defer:** radar-absorber performance is dominated by **geometry** (patch size, period,
  spacer) and the **sheet resistance**, which is a *tunable design parameter* (set by doping /
  patterning), not a fixed intrinsic property. Predicting an exact GHz conductivity for each new
  material buys little when the optimizer already tunes sheet resistance.
- **If a rigorous number is ever needed:** [AMSET](https://hackingmaterials.lbl.gov/amset/)
  (Boltzmann transport, BSD, pymatgen-integrated) computes electrical conductivity from a DFT band
  structure — but it needs DFT input, so run it only on the final shortlist. ML models for microwave
  *dielectric ceramics* (SISSO, Res-GCN) exist but target low-loss resonators, not absorbers.
- **Now:** classify metallic-vs-not via the band gap we already predict; leave sheet resistance to
  the optimizer.

## 3. DFT-in-the-loop validation  —  ✅ build (use MatterGen's own S.U.N. workflow)

**Best approach: MatterSim MLIP relax → energy-above-hull → S.U.N. filter → DFT only on the few.**
- **Workflow (MatterGen's published pipeline):** MLFF geometry optimization → filter by **S.U.N.**
  (Stable = E_hull < 0.1 eV/atom, Unique, Novel vs Alex-MP) → DFT double-relaxation → final
  properties. This removes spurious/non-equilibrium generated structures.
- **Tool:** [MatterSim](https://github.com/microsoft/mattersim) (`pip install mattersim`) — Microsoft's
  E(3)-equivariant universal MLIP (89 elements, 0–5000 K), for the fast relaxation + energy. DFT
  (VASP/Quantum ESPRESSO via `atomate2`) only for the handful that pass S.U.N.
- **Effort:** MEDIUM. MLIP stage fast on GPU; DFT stage compute-heavy, last-mile only.

## 4. Stability (energy-above-hull) + precursor model  —  ✅ build (shares #3's MLIP)

**Best approach: pymatgen convex hull for E_hull; SynthesisSimilarity for precursors.**
- **Stability:** relax with MatterSim (or matgl M3GNet/CHGNet) → compute formation energy →
  `pymatgen.PhaseDiagram` against Materials Project entries (`mp-api`) → **energy above hull**.
  This replaces our current raw-formation-energy proxy with the proper convex-hull metric.
- **Precursors:** [CederGroupHub/SynthesisSimilarity](https://github.com/CederGroupHub/SynthesisSimilarity)
  — He & Ceder, *Science Advances* 2023 ([arXiv 2302.02303](https://arxiv.org/abs/2302.02303)).
  Open code + 29,900 text-mined recipes; learns materials similarity and recommends precursor sets
  with **82% top-5 success**. Replaces our heuristic precursor map in `synthesis.py`.
- **Effort:** MEDIUM. Both open-source; stability shares the MatterSim setup from #3.

---

## Recommended build order

1. **GNNOpt optical n,k** — fixes the weakest numbers (the first-order n,k), runs on the pod today.
2. **MatterSim stability + S.U.N. validation** — turns raw formation energy into real convex-hull
   stability and filters junk structures (shared module for #3 + #4 stability).
3. **SynthesisSimilarity precursors** — real, literature-grounded synthesis routes in the dossier.
4. *(defer)* Radar conductivity → AMSET on final shortlist only.
5. *(last-mile)* Full DFT (VASP/QE) validation of the top candidates.

**Note on envs:** GNNOpt, MatterSim, and SynthesisSimilarity each have their own dependency stacks
(e3nn / Graphormer / TF respectively) — install each in its own venv (like `.venv-gen`) and hand off
via files, to keep the working pipeline intact.
