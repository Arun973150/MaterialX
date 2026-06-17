"""Predict refractive index n,k spectra for CIFs using pretrained GNNOpt models.

RUN ON THE POD, from inside the GNNOpt repo (so `utils.utils_model` + model/ + data/ resolve),
with the .venv-opt environment active:

    cd /workspace/GNNOpt
    cp /workspace/MaterialX/scripts/gnnopt_predict_nk.py .
    python gnnopt_predict_nk.py --cif-dir /workspace/runs/radar/cifs --out /workspace/runs/radar/gnnopt_nk.json

It reuses GNNOpt's exact featurization + the pretrained model_n/model_k checkpoints, derives the
training-set constants (scale_data, num_neighbors) from data/absorption_mp_data.pkl, and writes a
JSON: {id: {"energy_ev": [...251...], "n": [...], "k": [...]}}.

Note: GNNOpt's grid is 0-50 eV (electronic optical data) -> trustworthy for visible/NIR, coarse in
the thermal IR. The bridge uses it where it has resolution and falls back otherwise.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
from ase import Atom
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list
from mendeleev import element

from utils.utils_model import Network  # GNNOpt's e3nn network

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENERGY_MIN, ENERGY_MAX, NSTEP = 0.0, 50.0, 251
NEW_X = np.linspace(ENERGY_MIN, ENERGY_MAX, NSTEP)
R_MAX = 6.0
EM_DIM = 64
DATA_PKL = "data/absorption_mp_data.pkl"

# ---- full periodic-table encoding (exactly as GNNOpt) ----
type_encoding = {}
specie_mass, specie_dipole, specie_radius = [], [], []
for Z in range(1, 119):
    sp = Atom(Z)
    type_encoding[sp.symbol] = Z - 1
    specie_mass.append(sp.mass)
    dp = element(sp.symbol).dipole_polarizability
    specie_dipole.append(67.0 if dp is None else dp)
    rad = element(sp.symbol).covalent_radius_pyykko
    specie_radius.append(0.0 if rad is None else rad)
type_onehot = torch.eye(len(type_encoding))
mass_onehot = torch.diag(torch.tensor(specie_mass))
dipole_onehot = torch.diag(torch.tensor(specie_dipole))
radius_onehot = torch.diag(torch.tensor(specie_radius))


def build_data(atoms, r_max=R_MAX):
    symbols = list(atoms.symbols)
    positions = torch.from_numpy(atoms.positions.copy())
    lattice = torch.from_numpy(atoms.cell.array.copy()).unsqueeze(0)
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=atoms, cutoff=r_max, self_interaction=True)
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (
        positions[torch.from_numpy(edge_dst)]
        - positions[torch.from_numpy(edge_src)]
        + torch.einsum("ni,nij->nj", torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch])
    )
    return tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x_mass=mass_onehot[[type_encoding[s] for s in symbols]],
        x_dipole=dipole_onehot[[type_encoding[s] for s in symbols]],
        x_radius=radius_onehot[[type_encoding[s] for s in symbols]],
        z=type_onehot[[type_encoding[s] for s in symbols]],
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=np.around(edge_vec.norm(dim=1).numpy(), decimals=2),
    )


class MixingLinear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        w = torch.abs(self.weight) / (torch.sum(torch.abs(self.weight), dim=1, keepdim=True) + 1e-10)
        return F.linear(x, w)


class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, **kwargs):
        self.pool = False
        if kwargs["reduce_output"]:
            kwargs["reduce_output"] = False
            self.pool = True
        super().__init__(**kwargs)
        self.em_type = nn.Linear(in_dim, em_dim)
        self.em_mass = nn.Linear(in_dim, em_dim)
        self.em_dipole = nn.Linear(in_dim, em_dim)
        self.em_radius = nn.Linear(in_dim, em_dim)
        self.em_mixing = MixingLinear(3, 1)

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.z = F.relu(self.em_type(data.z))
        data.x_mass = F.relu(self.em_mass(data.x_mass))
        data.x_dipole = F.relu(self.em_dipole(data.x_dipole))
        data.x_radius = F.relu(self.em_radius(data.x_radius))
        tmp = torch.stack([data.x_mass, data.x_dipole, data.x_radius], dim=0)
        tmp2 = torch.permute(tmp, (1, 2, 0))
        data.x = torch.permute(self.em_mixing(tmp2), (2, 0, 1)).reshape(-1, EM_DIM)
        output = torch.relu(super().forward(data))
        if self.pool:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)
        return output


def _eps_to_nk(eps1, eps2):
    mod = np.sqrt(eps1 ** 2 + eps2 ** 2)
    return np.sqrt((mod + eps1) / 2), np.sqrt((mod - eps1) / 2)


def training_constants():
    """Compute scale_data (n,k) and num_neighbors from the training pickle."""
    import pandas as pd

    df = pd.read_pickle(DATA_PKL)
    n_max, k_max, counts = [], [], []
    for _, r in df.iterrows():
        n_raw, k_raw = _eps_to_nk(r["real_dielectric"], r["imag_dielectric"])
        n_max.append(np.interp(NEW_X, r["energies"], n_raw).max())
        k_max.append(np.interp(NEW_X, r["energies"], k_raw).max())
        src, _, _ = neighbor_list("ijS", a=r["structure"], cutoff=R_MAX, self_interaction=True)
        N = len(r["structure"])
        counts.extend(int((src == i).sum()) for i in range(N))
    return float(np.median(n_max)), float(np.median(k_max)), float(np.mean(counts))


def make_model(num_neighbors):
    return PeriodicNetwork(
        in_dim=118, em_dim=EM_DIM,
        irreps_in=f"{EM_DIM}x0e", irreps_out=f"{NSTEP}x0e", irreps_node_attr=f"{EM_DIM}x0e",
        layers=2, mul=32, lmax=2, max_radius=R_MAX, num_neighbors=num_neighbors, reduce_output=True,
    )


def load_model(which, num_neighbors):
    m = make_model(num_neighbors)
    state = torch.load(f"model/model_{which}_240406.torch", map_location=DEVICE)["state"]
    m.load_state_dict(state)
    m.pool = True
    return m.to(DEVICE).eval()


def predict(model, cif_files, scale):
    datas = [build_data(ase_read(f)) for f in cif_files]
    loader = tg.loader.DataLoader(datas, batch_size=16)
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append((model(batch.to(DEVICE)).cpu().numpy() * scale))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cif_files = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
    print(f"{len(cif_files)} CIFs from {args.cif_dir}")

    print("Computing training constants (scale_data, num_neighbors)...")
    scale_n, scale_k, num_neighbors = training_constants()
    print(f"  scale_n={scale_n:.3f}  scale_k={scale_k:.3f}  num_neighbors={num_neighbors:.2f}")

    model_n = load_model("n", num_neighbors)
    model_k = load_model("k", num_neighbors)
    print("Predicting n...")
    n_spec = predict(model_n, cif_files, scale_n)
    print("Predicting k...")
    k_spec = predict(model_k, cif_files, scale_k)

    result = {}
    for i, f in enumerate(cif_files):
        cid = os.path.splitext(os.path.basename(f))[0]
        result[cid] = {
            "energy_ev": NEW_X.tolist(),
            "n": [round(float(v), 4) for v in n_spec[i]],
            "k": [round(float(v), 4) for v in k_spec[i]],
        }
    with open(args.out, "w") as fh:
        json.dump(result, fh)
    print(f"Wrote n,k for {len(result)} materials -> {args.out}")


if __name__ == "__main__":
    main()
