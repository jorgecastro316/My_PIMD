#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob, math, warnings, time, hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../controller
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import expit  # numerically stable sigmoid
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import shutil, traceback
from contextlib import contextmanager
from pimd.utils import rr2nm, nm2rr

# Resolve symlink target → absolute project root
controller_path = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(controller_path))

# Add project_root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Optional deps for GPR mode ===

try:
    import torch, gpytorch
    _HAS_GPYTORCH = True
except ImportError as e:
    _HAS_GPYTORCH = False
    _GPYTORCH_IMPORT_ERROR = e


# === PIMD imports for PIMC pool mode ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../GRP_NE_PIMD/controller
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # .../GRP_NE_PIMD
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === PIMD imports for PIMC pool mode ===
try:
    from pimd.system import System
    from pimd.force_field import Force_Field
    from pimd.simulation import Simulation
    from pimd.thermostat import Thermostat
    from pimd.utils import read_input_file, parse_file_field
    _HAS_PIMD = True
    _PIMD_IMPORT_ERROR = None
except Exception as e:
    _HAS_PIMD = False
    _PIMD_IMPORT_ERROR = e

# === GLOBAL SEEDING ===
MASTER_SEED = 12345  # <-- choose once, all randomness flows from this

import random

def set_global_seed(seed=MASTER_SEED):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

set_global_seed(MASTER_SEED)

import os, torch
n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
torch.set_num_threads(n_threads)
print(f"[INFO] Training configured for {torch.get_num_threads()} threads "
      f"(SLURM_CPUS_PER_TASK={n_threads})")

# ---------- IO helpers ----------
def get_run_seed(run_id, master_seed=MASTER_SEED):
    # Combine master_seed and run_id into a string, hash it, and use as seed
    seed_str = f"{master_seed}_{run_id}"
    seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
    # Convert hash to integer seed (limit to 32 bits)
    return int(seed_hash[:8], 16)

def find_work_files(root, prefix_work):
    pattern = os.path.join(root, "**", f"{prefix_work}.work.out")
    return sorted(glob.glob(pattern, recursive=True))

def read_all_works(path):
    vals = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                vals.append(float(parts[-1]))
            except Exception:
                pass
    if not vals:
        raise ValueError(f"No work values in {path}")
    return vals

def read_multi_xyz(path):
    """
    Read a multi-frame XYZ file.

    Returns:
      arr: (n_frames, natoms*ndim) array
      natoms: int
      ndim: int
      symbols: list[str] (from first frame only)
    """
    frames = []
    symbols = None

    with open(path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            try:
                natoms = int(header.strip())
            except ValueError:
                raise ValueError(f"{path}: cannot parse natoms from '{header.strip()}'")

            comment = f.readline()
            coords = []
            frame_symbols = []
            for i in range(natoms):
                line = f.readline()
                if not line:
                    raise ValueError(f"{path}: unexpected EOF while reading atoms")
                parts = line.split()
                if symbols is None:   # only store for first frame
                    frame_symbols.append(parts[0])
                floats = [float(x) for x in parts[1:]]
                coords.extend(floats)

            if symbols is None:
                symbols = frame_symbols
                if len(symbols) != natoms:
                    raise ValueError(f"{path}: symbols length {len(symbols)} != natoms {natoms}")

            frames.append(coords)

    arr = np.array(frames, float)
    ndim = arr.shape[1] // natoms
    return arr, natoms, ndim, symbols

def farthest_point_sampling(Q, k, rng=None, start="auto"):
    """
    Q: (M, D) candidate vectors (flattened centroid positions from seed)
    k: number of points to select
    rng: np.random.Generator or RandomState
    start: "auto" | "first" | "random" | int (index)

    returns: idx (k,) integer indices into Q
    """
    if rng is None:
        rng = np.random.RandomState(MASTER_SEED)

    M = Q.shape[0]
    if k >= M:
        return np.arange(M)

    # pick initial center
    if start == "first":
        i0 = 0
    elif start == "random":
        i0 = int(rng.randint(0, M))
    elif start == "auto":
        mu = Q.mean(axis=0, keepdims=True)
        d2 = np.sum((Q - mu)**2, axis=1)
        i0 = int(np.argmax(d2))
    elif isinstance(start, int):
        if start < 0 or start >= M:
            raise ValueError("start index out of range")
        i0 = int(start)
    else:
        raise ValueError("invalid 'start' argument")

    idx = np.empty(k, dtype=int)
    idx[0] = i0

    dmin = np.sum((Q - Q[i0])**2, axis=1)

    for j in range(1, k):
        ij = int(np.argmax(dmin))
        idx[j] = ij
        dnew = np.sum((Q - Q[ij])**2, axis=1)
        dmin = np.minimum(dmin, dnew)

    return idx

def _flatten_samples(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # use centroid representation to keep dimensions consistent across bead counts
        arr = arr.mean(axis=1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f"Unexpected sample shape {arr.shape}")

def _describe(values):
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }

def _format_stats(stats):
    return ", ".join(f"{k}={v:.4f}" for k, v in stats.items())

def _sample_rows(mat, max_rows, rng=None):
    if max_rows is None or len(mat) <= max_rows:
        return mat
    rng = np.random.default_rng(rng)
    idx = rng.choice(len(mat), size=max_rows, replace=False)
    return mat[idx]

def _nn_to_seed_stats(pool_flat, seed_flat, max_pool=5000, max_seed=5000):
    if len(seed_flat) == 0 or len(pool_flat) == 0:
        return None
    pool_sub = _sample_rows(pool_flat, max_pool, rng=0)
    seed_sub = _sample_rows(seed_flat, max_seed, rng=1)
    d = cdist(pool_sub, seed_sub)
    mins = d.min(axis=1)
    return _describe(mins)

def _intra_pool_stats(pool_flat, max_rows=3000):
    if len(pool_flat) < 2:
        return None
    sub = _sample_rows(pool_flat, max_rows, rng=2)
    d = cdist(sub, sub)
    np.fill_diagonal(d, np.inf)
    mins = d.min(axis=1)
    return _describe(mins)

# ---------- BAR ----------

def bar_estimate(WF, WR, beta):
    WF = np.asarray(WF, float)
    WR = np.asarray(WR, float)
    nF, nR = len(WF), len(WR)
    if nF == 0 or nR == 0:
        raise ValueError("Need both forward and reverse samples for BAR.")
    c = np.log(nF / nR)

    def f(df):
        # use expit(-x) = 1/(1+exp(x)) to avoid overflow
        s1 = np.sum(expit(-(beta * (WF - df) - c)))
        s2 = np.sum(expit(-(beta * (WR + df) + c)))
        return s1 - s2

    # bracket based on min/max of actual data, not means
    lo = min(WF.min(), -WR.max()) - 5.0 / beta
    hi = max(WF.max(), -WR.min()) + 5.0 / beta

    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=2000)

def bar_ci(WF, WR, beta, df, confidence=0.95):
    WF = np.asarray(WF, float)
    WR = np.asarray(WR, float)
    nF, nR = len(WF), len(WR)
    c = np.log(nF / nR)

    p = expit(-(beta * (WF - df) - c))
    q = expit(-(beta * (WR + df) + c))

    # add epsilon to avoid zero Fisher info
    eps = 1e-16
    info = beta**2 * (np.sum(p * (1 - p)) + np.sum(q * (1 - q)))
    info = max(info, eps)

    z = norm.ppf(0.5 + confidence / 2.0)
    half = z / np.sqrt(info)
    return (df - half, df + half, 2 * half), info

class GroupedRBFKernel(gpytorch.kernels.Kernel):
    """
    RBF kernel with two group-wise lengthscales:
    - one for all q features
    - one for all p features
    """
    is_stationary = True

    def __init__(self, q_idx, p_idx, **kwargs):
        super().__init__(**kwargs)
        self.q_idx = torch.tensor(q_idx, dtype=torch.long)
        self.p_idx = torch.tensor(p_idx, dtype=torch.long)

        # raw lengthscales (log-space)
        self.register_parameter(
            name="raw_lengthscale_q",
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            name="raw_lengthscale_p",
            parameter=torch.nn.Parameter(torch.zeros(1))
        )

        self.register_constraint("raw_lengthscale_q", gpytorch.constraints.Positive())
        self.register_constraint("raw_lengthscale_p", gpytorch.constraints.Positive())

    @property
    def lengthscale_q(self):
        return self.raw_lengthscale_q_constraint.transform(self.raw_lengthscale_q)

    @property
    def lengthscale_p(self):
        return self.raw_lengthscale_p_constraint.transform(self.raw_lengthscale_p)

    def forward(self, x1, x2, diag=False, **params):
        x1_q, x2_q = x1[..., self.q_idx], x2[..., self.q_idx]
        x1_p, x2_p = x1[..., self.p_idx], x2[..., self.p_idx]

        if diag:
            # For diag, compute squared distances elementwise
            d2_q = (x1_q - x2_q).pow(2).sum(-1)
            d2_p = (x1_p - x2_p).pow(2).sum(-1)
        else:
            d2_q = (x1_q[:, None, :] - x2_q[None, :, :]).pow(2).sum(-1)
            d2_p = (x1_p[:, None, :] - x2_p[None, :, :]).pow(2).sum(-1)

        Kq = torch.exp(-0.5 * d2_q / self.lengthscale_q.pow(2))
        Kp = torch.exp(-0.5 * d2_p / self.lengthscale_p.pow(2))

        return Kq * Kp

class GroupedMaternKernel(gpytorch.kernels.Kernel):
    """
    Matern kernel with two group-wise lengthscales:
    - one for all q features
    - one for all p features
    Supports nu=0.5, 1.5, 2.5.
    """
    is_stationary = True

    def __init__(self, q_idx, p_idx, nu=2.5, **kwargs):
        super().__init__(**kwargs)
        self.q_idx = torch.tensor(q_idx, dtype=torch.long)
        self.p_idx = torch.tensor(p_idx, dtype=torch.long)
        self.nu = nu

        # raw lengthscales
        self.register_parameter(
            name="raw_lengthscale_q",
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            name="raw_lengthscale_p",
            parameter=torch.nn.Parameter(torch.zeros(1))
        )

        self.register_constraint("raw_lengthscale_q", gpytorch.constraints.Positive())
        self.register_constraint("raw_lengthscale_p", gpytorch.constraints.Positive())

    @property
    def lengthscale_q(self):
        return self.raw_lengthscale_q_constraint.transform(self.raw_lengthscale_q)

    @property
    def lengthscale_p(self):
        return self.raw_lengthscale_p_constraint.transform(self.raw_lengthscale_p)

    def forward(self, x1, x2, diag=False, **params):
        x1_q, x2_q = x1[..., self.q_idx], x2[..., self.q_idx]
        x1_p, x2_p = x1[..., self.p_idx], x2[..., self.p_idx]

        if diag:
            d_q = torch.norm((x1_q - x2_q) / self.lengthscale_q, dim=-1)
            d_p = torch.norm((x1_p - x2_p) / self.lengthscale_p, dim=-1)
            d = torch.sqrt(d_q**2 + d_p**2 + 1e-12)   # shape [N]
        else:
            d_q = torch.cdist(x1_q / self.lengthscale_q, x2_q / self.lengthscale_q, p=2)
            d_p = torch.cdist(x1_p / self.lengthscale_p, x2_p / self.lengthscale_p, p=2)
            d = torch.sqrt(d_q**2 + d_p**2 + 1e-12)   # shape [N, M]

        sqrt_2nu_d = torch.sqrt(torch.tensor(2.0 * self.nu, device=d.device)) * d
        if self.nu == 0.5:
            k = torch.exp(-sqrt_2nu_d)
        elif self.nu == 1.5:
            k = (1.0 + sqrt_2nu_d) * torch.exp(-sqrt_2nu_d)
        elif self.nu == 2.5:
            k = (1.0 + sqrt_2nu_d + (sqrt_2nu_d**2)/3.0) * torch.exp(-sqrt_2nu_d)
        else:
            raise NotImplementedError("Only nu=0.5,1.5,2.5 supported")

        return k


class PerModeRBFKernel(gpytorch.kernels.Kernel):
    """
    RBF kernel with independent lengthscales per normal-mode block
    for both q and p features (falls back to single block for centroid/raw).
    """
    is_stationary = True

    def __init__(self, q_slices, p_slices, **kwargs):
        super().__init__(**kwargs)
        self.q_slices = q_slices or []
        self.p_slices = p_slices or []
        if self.q_slices:
            self.register_parameter(
                name="raw_lengthscale_q",
                parameter=torch.nn.Parameter(torch.zeros(len(self.q_slices)))
            )
            self.register_constraint("raw_lengthscale_q", gpytorch.constraints.Positive())
        else:
            self.raw_lengthscale_q = None
        if self.p_slices:
            self.register_parameter(
                name="raw_lengthscale_p",
                parameter=torch.nn.Parameter(torch.zeros(len(self.p_slices)))
            )
            self.register_constraint("raw_lengthscale_p", gpytorch.constraints.Positive())
        else:
            self.raw_lengthscale_p = None

    @property
    def lengthscale_q(self):
        if self.raw_lengthscale_q is None:
            return None
        return self.raw_lengthscale_q_constraint.transform(self.raw_lengthscale_q)

    @property
    def lengthscale_p(self):
        if self.raw_lengthscale_p is None:
            return None
        return self.raw_lengthscale_p_constraint.transform(self.raw_lengthscale_p)

    def _accumulate(self, x1, x2, slices, lscales, diag):
        if not slices or lscales is None:
            if diag:
                return torch.zeros(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
            return torch.zeros((x1.size(0), x2.size(0)), device=x1.device, dtype=x1.dtype)

        total = None
        for idx, (start, end) in enumerate(slices):
            block1 = x1[..., start:end]
            block2 = x2[..., start:end]
            if diag:
                diff = (block1 - block2).pow(2).sum(-1)
            else:
                diff = (block1[:, None, :] - block2[None, :, :]).pow(2).sum(-1)
            contrib = diff / (lscales[idx] ** 2 + 1e-12)
            total = contrib if total is None else total + contrib
        return total

    def forward(self, x1, x2, diag=False, **params):
        if diag and x2 is None:
            x2 = x1
        d2_q = self._accumulate(x1, x2, self.q_slices, self.lengthscale_q, diag)
        d2_p = self._accumulate(x1, x2, self.p_slices, self.lengthscale_p, diag)
        d2_total = d2_q + d2_p
        return torch.exp(-0.5 * d2_total)

# ==================================================
# GPytorch Exact GP Model
# ==================================================
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 ard=False, input_dim=None, grouped=False, kernel="rbf",
                 mode_slices=None):
        super().__init__(train_x, train_y, likelihood)
        if input_dim is None:
            raise ValueError("ExactGPModel requires input_dim to define LinearMean.")

        self.mean_module = gpytorch.means.LinearMean(input_dim)

        # --- Kernel selection ---
        if kernel == "rbf":
            if grouped:
                q_idx = list(range(0, input_dim // 2))
                p_idx = list(range(input_dim // 2, input_dim))
                base_kernel = GroupedRBFKernel(q_idx=q_idx, p_idx=p_idx)
            else:
                base_kernel = gpytorch.kernels.RBFKernel(
                    ard_num_dims=input_dim if ard else None
                )

        elif kernel == "matern15":
            if grouped:
                q_idx = list(range(0, input_dim // 2))
                p_idx = list(range(input_dim // 2, input_dim))
                base_kernel = GroupedMaternKernel(q_idx=q_idx, p_idx=p_idx, nu=1.5)
            else:
                base_kernel = gpytorch.kernels.MaternKernel(
                    nu=1.5, ard_num_dims=input_dim if ard else None
                )

        elif kernel == "matern25":
            if grouped:
                q_idx = list(range(0, input_dim // 2))
                p_idx = list(range(input_dim // 2, input_dim))
                base_kernel = GroupedMaternKernel(q_idx=q_idx, p_idx=p_idx, nu=2.5)
            else:
                base_kernel = gpytorch.kernels.MaternKernel(
                    nu=2.5, ard_num_dims=input_dim if ard else None
                )
        elif kernel == "permode":
            if mode_slices is None:
                raise ValueError("Per-mode kernel requires mode_slices.")
            base_kernel = PerModeRBFKernel(
                q_slices=mode_slices.get("q"),
                p_slices=mode_slices.get("p")
            )

        else:
            raise ValueError(f"Unknown kernel type: {kernel}")

        # Wrap kernel in a ScaleKernel
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        # Store for debugging/logging
        self.kernel_type = kernel
        self.grouped = grouped

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class WorkGPRTorch:
    def __init__(self, feature_mode="centroid", ard=False, grouped=False, kernel="rbf", n_iter=300,
                 lr=0.1, noise_min=1e-6, seed=123,
                 device="cpu", scale_work=False, make_plots=False,
                 jitter=1e-3, cg_max_iters=2000, precond_size=200):
        self.feature_mode = feature_mode
        self.ard = ard
        self.grouped = grouped
        self.n_iter = n_iter
        self.lr = lr
        self.noise_min = noise_min
        self.seed = seed
        self.device = torch.device(device)
        self.scale_work = scale_work
        self.make_plots = make_plots
        self.gp_forward = None
        self.gp_reverse = None
        self.kernel = kernel
        self.jitter = jitter
        self.cg_max_iters = cg_max_iters
        self.precond_size = precond_size

    @contextmanager
    def _gp_settings(self):
        with gpytorch.settings.cholesky_jitter(self.jitter), \
             gpytorch.settings.max_cg_iterations(self.cg_max_iters), \
             gpytorch.settings.max_preconditioner_size(self.precond_size):
            yield

    def _apply_feature_mode(self, q, p, return_meta=False):
        """
        Apply feature mode consistently for both training and prediction.
        """
        fm = self.feature_mode
        meta = {}

        if fm.startswith("nmodes:"):
            _, mode_str = fm.split(":")
            if mode_str.strip().lower() == "all":
                modes = list(range(q.shape[1]))
            else:
                modes = [int(m) for m in mode_str.split(",")]
            q, p = select_normal_modes(q, p, modes)
            meta["mode_count"] = len(modes)

        elif fm == "centroid":
            q, p = _reduce_beads(q, p, centroid=True)
            meta["mode_count"] = 1

        elif fm == "raw":
            q, p = _reduce_beads(q, p, centroid=False)
            meta["mode_count"] = 1

        else:
            raise ValueError(f"[GPR] Unknown feature_mode '{fm}'")

        if return_meta:
            return q, p, meta
        return q, p

    def _compute_mode_slices(self, feature_dim, mode_count):
        if mode_count <= 1:
            return [(0, feature_dim)]
        if feature_dim % mode_count != 0:
            raise ValueError(f"[GPR] feature_dim={feature_dim} not divisible by mode_count={mode_count}")
        block = feature_dim // mode_count
        return [(i * block, (i + 1) * block) for i in range(mode_count)]

    def _fit(self, q, p, w, label="F", round_num=0):
        y_raw = np.asarray(w).ravel().reshape(-1, 1)

        # --- Apply feature mode ---
        q, p, meta = self._apply_feature_mode(q, p, return_meta=True)

        # --- Scale features ---
        scaler_q = StandardScaler()
        scaler_p = StandardScaler()
        q_scaled = scaler_q.fit_transform(q)
        p_scaled = scaler_p.fit_transform(p)
        X = np.concatenate([q_scaled, p_scaled], axis=1)

        mode_count = meta.get("mode_count", 1)
        q_slices = self._compute_mode_slices(q.shape[1], mode_count)
        p_slices = self._compute_mode_slices(p.shape[1], mode_count)
        mode_slices = {"q": q_slices, "p": p_slices}

        # --- Scale work if needed ---
        if self.scale_work:
            scaler_w = StandardScaler()
            y = scaler_w.fit_transform(w.reshape(-1, 1)).ravel()
        else:
            scaler_w = None
            y = w.ravel()

        # --- Train/val split ---
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        train_x = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        val_x = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        val_y = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(self.noise_min)
        ).to(self.device)

        model = ExactGPModel(
            train_x, train_y, likelihood,
            ard=self.ard,
            input_dim=X.shape[1],
            grouped=self.grouped,
            kernel=self.kernel,
            mode_slices=mode_slices if self.kernel == "permode" else None
        ).to(self.device)

        print(f"[TRAIN] Using {torch.get_num_threads()} PyTorch threads for GP training.")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        def _fmt_ls(ls_tensor):
            if ls_tensor is None:
                return "n/a"
            arr = ls_tensor.detach().cpu().numpy().ravel()
            if arr.size == 1:
                return f"{arr[0]:.4f}"
            return "[" + ", ".join(f"{v:.3f}" for v in arr) + "]"

        with self._gp_settings():
            model.train(); likelihood.train()
            for i in range(self.n_iter):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
                if (i + 1) % 50 == 0 or i == 0 or i == self.n_iter - 1:
                    noise = likelihood.noise.item()
                    base_kernel = model.covar_module.base_kernel
                    if hasattr(base_kernel, "lengthscale_q"):
                        ls_q = _fmt_ls(base_kernel.lengthscale_q)
                        ls_p = _fmt_ls(base_kernel.lengthscale_p)
                        print(f"[GPR:{label}] Iter {i+1}/{self.n_iter}, "
                              f"Loss={loss.item():.4f}, "
                              f"Lq={ls_q}, Lp={ls_p}, noise={noise:.3e}")
                    else:
                        ls = base_kernel.lengthscale.detach().cpu().numpy()
                        print(f"[GPR:{label}] Iter {i+1}/{self.n_iter}, "
                              f"Loss={loss.item():.4f}, "
                              f"lengthscale={ls}, noise={noise:.3e}")

        # --- Validation ---
        model.eval(); likelihood.eval()
        with self._gp_settings(), torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_val = likelihood(model(val_x))

        mu_scaled = pred_val.mean.cpu().numpy().reshape(-1, 1)
        std_scaled = pred_val.stddev.cpu().numpy().reshape(-1, 1)

        if scaler_w is not None:
            mu = scaler_w.inverse_transform(mu_scaled).ravel()
            std = std_scaled.ravel() * scaler_w.scale_[0]
            y_val_true = scaler_w.inverse_transform(val_y.cpu().numpy().reshape(-1, 1)).ravel()
        else:
            mu = mu_scaled.ravel()
            std = std_scaled.ravel()
            y_val_true = val_y.cpu().numpy().ravel()

        rmse = np.sqrt(mean_squared_error(y_val_true, mu))
        r2 = r2_score(y_val_true, mu)
        coverage = np.mean((y_val_true >= mu - 2*std) & (y_val_true <= mu + 2*std))

        print(f"[GPR:{label}] Val RMSE={rmse:.4f}, R²={r2:.4f}, Coverage={coverage:.2f}")

        return {
            "model": model,
            "likelihood": likelihood,
            "scaler_q": scaler_q,
            "scaler_p": scaler_p,
            "scaler_w": scaler_w,
            "train_w_std": y_raw.std(),
            "rmse": rmse,
            "r2": r2,
            "coverage": coverage,
        }

    def fit_forward(self, q, p, w, round_num=0):
        self.gp_forward = self._fit(q, p, w, label="F", round_num=round_num)
        return self.gp_forward

    def fit_reverse(self, q, p, w, round_num=0):
        self.gp_reverse = self._fit(q, p, w, label="R", round_num=round_num)
        return self.gp_reverse


    def _predict(self, block, q, p, label="F", round_num=0):
        # NOTE: q, p are already preprocessed outside

        # --- Scale with training scalers ---
        q_feat = block["scaler_q"].transform(q)
        p_feat = block["scaler_p"].transform(p)
        X = np.concatenate([q_feat, p_feat], axis=1)
        test_x = torch.tensor(X, dtype=torch.float32, device=self.device)

        block["model"].eval(); block["likelihood"].eval()
        with self._gp_settings(), torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = block["likelihood"](block["model"](test_x))

        mu_scaled = pred.mean.cpu().numpy().reshape(-1, 1)
        std_scaled = pred.stddev.cpu().numpy().reshape(-1, 1)

        if block["scaler_w"] is not None:
            mu = block["scaler_w"].inverse_transform(mu_scaled).ravel()
            std = std_scaled.ravel() * block["scaler_w"].scale_[0]
        else:
            mu = mu_scaled.ravel()
            std = std_scaled.ravel()

        print(f"[GPR:{label}] Pred mean stats: mean={mu.mean():.4f}, std={mu.std():.4f}, "
            f"min={mu.min():.4f}, max={mu.max():.4f}")
        print(f"[GPR:{label}] Pred sigma stats: mean={std.mean():.4f}, std={std.std():.4f}, "
            f"min={std.min():.4f}, max={std.max():.4f}")

        return mu, std


    def predict_forward(self, q, p): return self._predict(self.gp_forward, q, p, label="F")
    def predict_reverse(self, q, p): return self._predict(self.gp_reverse, q, p, label="R")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.gp_forward, os.path.join(path, "forward.pt"))
        torch.save(self.gp_reverse, os.path.join(path, "reverse.pt"))
        # Save scalers
        joblib.dump(self.gp_forward["scaler_q"], os.path.join(path, "scaler_q_forward.pkl"))
        joblib.dump(self.gp_forward["scaler_p"], os.path.join(path, "scaler_p_forward.pkl"))
        joblib.dump(self.gp_forward["scaler_w"], os.path.join(path, "scaler_w_forward.pkl"))
        joblib.dump(self.gp_reverse["scaler_q"], os.path.join(path, "scaler_q_reverse.pkl"))
        joblib.dump(self.gp_reverse["scaler_p"], os.path.join(path, "scaler_p_reverse.pkl"))
        joblib.dump(self.gp_reverse["scaler_w"], os.path.join(path, "scaler_w_reverse.pkl"))

    @staticmethod
    def load(path, device="cpu"):
        obj = WorkGPRTorch(device=device)
        obj.gp_forward = torch.load(os.path.join(path, "forward.pt"), map_location=device, weights_only=False)
        obj.gp_reverse = torch.load(os.path.join(path, "reverse.pt"), map_location=device, weights_only=False)
        # Reload scalers
        obj.gp_forward["scaler_q"] = joblib.load(os.path.join(path, "scaler_q_forward.pkl"))
        obj.gp_forward["scaler_p"] = joblib.load(os.path.join(path, "scaler_p_forward.pkl"))
        obj.gp_forward["scaler_w"] = joblib.load(os.path.join(path, "scaler_w_forward.pkl"))
        obj.gp_reverse["scaler_q"] = joblib.load(os.path.join(path, "scaler_q_reverse.pkl"))
        obj.gp_reverse["scaler_p"] = joblib.load(os.path.join(path, "scaler_p_reverse.pkl"))
        obj.gp_reverse["scaler_w"] = joblib.load(os.path.join(path, "scaler_w_reverse.pkl"))
        return obj


def rank_forward_uncertainty(stdF, topk=100):
    idx = np.argsort(-stdF)          # highest uncertainty first
    return idx[:min(topk, len(idx))]

def rank_reverse_uncertainty(stdR, topk=100):
    idx = np.argsort(-stdR)
    return idx[:min(topk, len(idx))]

def _load_train_npz(path):
    d=np.load(path)
    for k in ["qF","pF","wF","qR","pR","wR"]:
        if k not in d: raise KeyError(f"Training NPZ missing key: {k}")
    return {k:d[k] for k in ["qF","pF","wF","qR","pR","wR"]}

def _load_cand_npz(path):
    d=np.load(path)
    for k in ["q","p"]:
        if k not in d: raise KeyError(f"Candidates NPZ missing key: {k}")
    return {k:d[k] for k in ["q","p"]}

def _reduce_beads(q, p, centroid=True):
    """
    Reduce multi-bead arrays to centroid (default) or flatten all beads.

    Parameters
    ----------
    q, p : np.ndarray
        Shapes:
          - (n_samples, nbeads, natoms*ndim)
          - (n_samples, natoms*ndim)
    centroid : bool
        If True -> average across beads (axis=1).
        If False -> flatten beads into features.
    """
    if q.ndim == 3:
        if centroid:
            q = q.mean(axis=1)
            p = p.mean(axis=1)
        else:
            q = q.reshape(q.shape[0], -1)
            p = p.reshape(p.shape[0], -1)
    elif q.ndim == 2:
        # Already reduced (centroid or 1 bead)
        pass
    else:
        raise ValueError(f"Unexpected q shape {q.shape}")
    return q, p

def select_normal_modes(q, p, modes):
    """
    Transform bead coordinates/momenta to normal modes and select subset.

    Parameters
    ----------
    q, p : np.ndarray
        shape (n_samples, nbeads, D)
    modes : list[int]
        indices of normal modes to keep

    Returns
    -------
    q_sel, p_sel : np.ndarray
        shape (n_samples, len(modes)*D)
    """
    if q.ndim != 3 or p.ndim != 3:
        raise ValueError(f"Expected bead arrays with ndim=3, got q={q.shape}, p={p.shape}")
    n_samples, nbeads, D = q.shape
    q_nm = np.array([rr2nm(q[i].T).T for i in range(n_samples)])  # (n_samples, nbeads, D)
    p_nm = np.array([rr2nm(p[i].T).T for i in range(n_samples)])
    q_sel = q_nm[:, modes, :].reshape(n_samples, -1)
    p_sel = p_nm[:, modes, :].reshape(n_samples, -1)
    return q_sel, p_sel

def perturb_ring_polymer_modes(system, sigma, max_mode=None):
    """
    Inject Gaussian noise into non-centroid normal modes of the system positions.

    Parameters
    ----------
    system : pimd.system.System
        System whose bead coordinates will be perturbed.
    sigma : float
        Standard deviation of the Gaussian perturbation. <=0 disables perturbation.
    max_mode : int | None
        Highest normal mode index (inclusive) to perturb. None -> all non-centroid modes.
    """
    if sigma is None or sigma <= 0:
        return False
    positions = getattr(system, "positions", None)
    if positions is None or positions.ndim != 2:
        return False
    q_nm = rr2nm(positions.T).T  # (nbeads, D)
    n_modes = q_nm.shape[0]
    start = 1  # skip centroid mode
    stop = n_modes if max_mode is None else min(n_modes, int(max_mode) + 1)
    if stop <= start:
        return False
    rng = getattr(system, "rng", None)
    if rng is None:
        rng = np.random.RandomState(MASTER_SEED)
    noise_shape = q_nm[start:stop].shape
    q_nm[start:stop] += rng.normal(0.0, sigma, size=noise_shape)
    perturbed = nm2rr(q_nm.T).T
    system.set_positions(perturbed)
    system.q_centroid = np.mean(system.positions, axis=0, keepdims=True)
    return True

def _ensure_frame_nbeads(frame, target_beads, label="frame"):
    """
    Ensure single configuration has bead axis of length target_beads.
    Allows expanding centroid-only frames to full rings (via tiling) and
    collapsing arbitrary bead counts to centroid (via averaging).
    """
    arr = np.asarray(frame)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"[SHAPE] {label}: expected 1D/2D array, got shape {arr.shape}")
    beads_now = arr.shape[0]
    if beads_now == target_beads:
        return arr
    if target_beads == 1:
        return arr.mean(axis=0, keepdims=True)
    if beads_now == 1:
        return np.tile(arr, (target_beads, 1))
    raise ValueError(f"[SHAPE] Cannot coerce {label} from {beads_now}→{target_beads} beads")

def _ensure_pool_nbeads(pool_arr, target_beads, label="pool"):
    """
    Ensure candidate pools have bead axis length target_beads.
    Accepts (N, beads, D) or (N, D) arrays.
    """
    arr = np.asarray(pool_arr)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError(f"[SHAPE] {label}: expected 2D/3D array, got shape {arr.shape}")
    beads_now = arr.shape[1]
    if beads_now == target_beads:
        return arr
    if target_beads == 1:
        return arr.mean(axis=1, keepdims=True)
    if beads_now == 1:
        return np.tile(arr, (1, target_beads, 1))
    raise ValueError(f"[SHAPE] Cannot coerce {label} from {beads_now}→{target_beads} beads")

def preprocess_features(q, p, model, label=""):
    """
    Transform candidate features to match model.feature_mode and check consistency.
    """
    fm = getattr(model, "feature_mode", "centroid")

    if fm == "centroid":
        q, p = _reduce_beads(q, p, centroid=True)

    elif fm == "all":
        q, p = _reduce_beads(q, p, centroid=False)

    elif fm.startswith("nmodes:"):
        mode_str = fm.split(":")[1]
        if mode_str == "all":
            modes = list(range(q.shape[1]))  # all beads
        else:
            modes = [int(m) for m in mode_str.split(",")]
        q, p = select_normal_modes(q, p, modes)

    else:
        raise ValueError(f"[GPR] Unknown feature_mode '{fm}'")

    # --- Safety check vs model dimensions ---
    n_features = q.shape[1] + p.shape[1]
    model_ls = None
    try:
        model_ls = model.gp_forward.covar_module.base_kernel.lengthscale
        model_dim = model_ls.numel()
    except Exception:
        model_dim = "unknown"

    print(f"[CHECK:{label}] feature_mode={fm} | features={n_features} | model_lengthscales={model_dim}")

    if model_dim != "unknown" and n_features != model_dim:
        raise ValueError(
            f"[GPR:{label}] Inconsistent feature dimensions! "
            f"Features={n_features}, model expects={model_dim}"
        )

    return q, p


# ===================== MAIN =====================
def main():

    args=sys.argv[1:]
    if "--pimc-pool" in args:
        if not _HAS_PIMD:
            raise SystemExit(f"Failed to import pimd: {type(_PIMD_IMPORT_ERROR).__name__}: {_PIMD_IMPORT_ERROR}")

        out_pimc_npz = args[args.index("--pimc-pool")+1]
        N = int(args[args.index("--N")+1])
        nbeads = int(args[args.index("--nbeads")+1])
        seed_npz = args[args.index("--seed-npz")+1]
        direction = args[args.index("--direction")+1].lower()
        print(f"[PIMC] Seed snapshot from {seed_npz}")
        data = np.load(seed_npz, allow_pickle=True)

        if direction == "forward":
            q0 = data["qF"][-1]
        elif direction == "reverse":
            q0 = data["qR"][-1]
        else:
            raise ValueError("direction must be forward|reverse")

        symbols = data["symbols"].tolist()

         # === Read input ===
        input_file = args[args.index("--input-file")+1] if "--input-file" in args else "input.txt"

        print(f"[PIMC] Reading input file: {input_file}")
        inp = read_input_file(input_file)

        #Initialze random number generator
        master_seed = inp.get("seed", 420)
        run_id = inp.get("run_id", 0)
        run_seed = get_run_seed(run_id, master_seed)
        rng = np.random.RandomState(run_seed)

        #Set the system
        n_beads = int(args[args.index("--nbeads")+1] if "--nbeads" in args else inp["n_beads"])

        system_temperature = float(inp["temperature"])
        print(f"System temperature: {system_temperature} K")
        simulation_temperature = system_temperature * n_beads  # PIMD temperature
        print(f"Simulation temperature: {simulation_temperature} K")

        # Build System from centroid snapshot
        system = System(
            geometry_file=None,
            n_beads=n_beads,
            rng=rng,
            temperature=simulation_temperature,
            centroid_q=q0,
            symbols=symbols
        )

        print(f"System initialized with {system.n_atoms} atoms, {system.n_beads} beads, and {system.n_dim} dimensions.")
        print(f"System initialized with {system.n_atoms} atoms, {system.n_beads} beads, and {system.n_dim} dimensions.")
        print(f"Initial positions: {system.positions}",np.shape(system.positions))
        print(f"Initial centroid positions: {system.q_centroid}",np.shape(system.q_centroid))
        print(f"Initial masses: {system.masses}",np.shape(system.masses))
        print(f"Initial forces: {system.forces}",np.shape(system.forces))

        #Set the force field
        forces = inp["forces"]
        force_field = Force_Field(forces,direction=direction)

        #Set a dummy thermostat
        thermostat_type = inp.get("thermostat", None)
        tau = inp.get("tau", 1)
        thermostat = Thermostat(simulation_temperature,tau,thermostat_type,rng)

        #Set the simulation.
        timestep = inp["timestep"] #For PIMC this is just a dummy value
        prefix = inp.get("prefix", "PIMC_pool")
         # Use zero output frequency to avoid writing files
        stride_out = 0
        stride_pos = 0 # Save every frame
        stride_vel = 0
        stride_log = 0
        simulation = Simulation(system,simulation_temperature,timestep,force_field,thermostat,prefix=prefix,stride_out=stride_out,stride_pos=stride_pos,stride_vel=stride_vel,stride_log=stride_log)
        print(f"[PIMC] Generating {N} samples from {direction} seed with {nbeads} beads")
            # --- Handle output flags ---
        out_text = True
        if "--no-text" in args:
            out_text = False
        if "--out-text" in args:
            out_text = True

        start_time = time.perf_counter()
        #simulation.run_PIMC(N,step_size=0.1,out_npz=out_pimc_npz,out_text=out_text,gen_velocities=True)
        #simulation.run_PIMC_adaptive(N,out_npz=out_pimc_npz,out_text=out_text,gen_velocities=True)
        simulation.run_PIMC_enhanced(N,out_npz=out_pimc_npz,out_text=out_text,gen_velocities=True,enable_bisection=False,p_single=0.25,p_displacement=0.75)
        end_time = time.perf_counter()
        total_seconds = end_time - start_time
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"\nSimulation completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        return

    if "--pimc-fps-pool" in args:
        if not _HAS_PIMD:
            raise SystemExit(f"Failed to import pimd: {type(_PIMD_IMPORT_ERROR).__name__}: {_PIMD_IMPORT_ERROR}")

        out_npz_all = args[args.index("--pimc-fps-pool")+1]
        N_total     = int(args[args.index("--N")+1])
        nbeads_cli  = int(args[args.index("--nbeads")+1])
        seed_npz    = args[args.index("--seed-npz")+1]
        direction   = args[args.index("--direction")+1].lower()
        X           = int(args[args.index("--X")+1]) if "--X" in args else 4
        fps_start   = args[args.index("--fps-start")+1] if "--fps-start" in args else "auto"
        use_enhanced = "--enhanced" in args
        use_adaptive = "--adaptive" in args
        state_file  = args[args.index("--state-file")+1] if "--state-file" in args else None
        resume_state = None
        branch_ids_arr = None
        if state_file and os.path.exists(state_file):
            try:
                with np.load(state_file, allow_pickle=True) as st:
                    q_resume = np.array(st["q_last"])
                    stored_ids = np.array(st["branch_ids"]) if "branch_ids" in st else None
                    nbeads_state = int(st["nbeads"]) if "nbeads" in st else nbeads_cli
                    natoms_state = int(st["natoms"]) if "natoms" in st else None
                    ndim_state = int(st["ndim"]) if "ndim" in st else None
            except Exception as exc:
                print(f"[PIMC-FPS] Failed to load state file {state_file}: {exc}")
                q_resume = None
            if q_resume is not None:
                if q_resume.shape[0] != X:
                    print(f"[PIMC-FPS] State file has {q_resume.shape[0]} branches but X={X}; ignoring saved state.")
                elif nbeads_state != nbeads_cli:
                    raise SystemExit(f"[PIMC-FPS] State file nbeads={nbeads_state} "
                                     f"does not match requested nbeads={nbeads_cli}.")
                else:
                    branch_ids_arr = stored_ids if stored_ids is not None else np.arange(X)
                    branch_ids_arr = np.array(branch_ids_arr, dtype=int)
                    if branch_ids_arr.shape[0] != X:
                        print(f"[PIMC-FPS] State file branch_ids mismatch; expected {X} entries. "
                              f"Ignoring saved state.")
                        resume_state = None
                        branch_ids_arr = None
                    else:
                        resume_state = {
                            "q": q_resume,
                            "natoms": natoms_state,
                            "ndim": ndim_state,
                            "nbeads": nbeads_state
                        }
                        print(f"[PIMC-FPS] Loaded branch states from {state_file}")
        elif state_file:
            print(f"[PIMC-FPS] No previous state file at {state_file}; starting new branches.")

        def _get_arg(flag, cast=float, default=None):
            if flag in args:
                try:
                    return cast(args[args.index(flag)+1])
                except Exception as exc:
                    raise SystemExit(f"[PIMC-FPS] Failed to parse {flag}: {exc}") from exc
            return default

        # Velocity generation defaults to ON (can be disabled explicitly)
        gen_vel = True
        if "--no-gen-vel" in args:
            gen_vel = False
        elif "--gen-vel" in args:
            gen_vel = True

        # Optional plain text toggle
        out_text = ("--out-text" in args) or ("--no-text" not in args)

        # Normal-mode randomization controls
        nm_noise = _get_arg("--fps-nm-noise", float, 0.0)
        nm_max_mode = _get_arg("--fps-nm-max-mode", int, None)

        # Plain PIMC knobs
        curr_plain_step = _get_arg("--fps-step", float, 0.1)

        # Enhanced-mode knobs (defaults match previous hard-coded values)
        curr_step_single = _get_arg("--fps-step-single", float, 0.05)
        curr_step_disp   = _get_arg("--fps-step-disp", float, 0.05)
        curr_p_single = _get_arg("--fps-p-single", float, 0.4)
        curr_p_disp   = _get_arg("--fps-p-displacement", float, 0.50)
        curr_p_bis    = _get_arg("--fps-p-bisection", float, 0.10)
        enable_bisection = "--fps-disable-bisection" not in args
        carry_steps = "--fps-no-carry-steps" not in args
        auto_adjust = "--fps-no-auto-adjust" not in args
        accept_target_lo = _get_arg("--fps-target-accept-low", float, 0.30)
        accept_target_hi = _get_arg("--fps-target-accept-high", float, 0.60)
        auto_scale_up = _get_arg("--fps-step-scale-up", float, 1.35)
        auto_scale_down = _get_arg("--fps-step-scale-down", float, 0.80)
        auto_step_min = _get_arg("--fps-step-min", float, 1e-4)
        auto_step_max = _get_arg("--fps-step-max", float, 2.0)

        print(f"[PIMC-FPS] Seed file: {seed_npz} | direction={direction} | X={X} | total N={N_total}")

        data = np.load(seed_npz, allow_pickle=True)

        # Pick direction arrays
        if direction == "forward":
            Qseed = data["qF"]  # shape: (frames, 1, D)
        elif direction == "reverse":
            Qseed = data["qR"]
        else:
            raise ValueError("direction must be forward|reverse")

        # Symbols from seed
        if "symbols" in data:
            symbols = data["symbols"].tolist()
        else:
            raise KeyError("[PIMC-FPS] Seed file missing 'symbols' key — required to run SPIMD later.")

        # Handle centroid vs full-ring initialization
        if Qseed.ndim == 3:  # (frames, nbeads, D)
            if "--init-centroid" in args:
                # Average over beads → centroid
                Qflat = Qseed.mean(axis=1)        # (frames, D)
            else:
                # Keep all beads
                Qflat = Qseed                     # (frames, nbeads, D)
        elif Qseed.ndim == 2:  # old centroid-only case
            Qflat = Qseed if "--init-centroid" in args else Qseed[:, None, :]
        else:
            raise ValueError(f"[PIMC-FPS] Unexpected Qseed shape {Qseed.shape}")

        print(f"[PIMC-FPS] Seed frames: {Qflat.shape}")


        # Read input file
        input_file = args[args.index("--input-file")+1] if "--input-file" in args else "input.txt"
        inp = read_input_file(input_file)

        # Global RNG
        master_seed = inp.get("seed", 420)
        run_id = inp.get("run_id", 0)
        base_seed = get_run_seed(run_id, master_seed)


        # Beads setup
        n_beads = int(args[args.index("--nbeads")+1] if "--nbeads" in args else inp["n_beads"])
        if n_beads != nbeads_cli:
            print(f"[WARN] using n_beads={n_beads} (CLI {nbeads_cli}, input {inp.get('n_beads')})")

        T_sys = float(inp["temperature"])
        T_sim = T_sys * n_beads
        print(f"[PIMC-FPS] System T={T_sys} K | Simulation T={T_sim} K")

        # Force field and thermostat
        force_field = Force_Field(inp["forces"], direction=direction)
        thermostat_type = inp.get("thermostat", None)
        tau = inp.get("tau", 1.0)
        timestep = inp["timestep"]
        prefix = inp.get("prefix", "PIMC_pool_FPS")

        branch_configs = []
        if resume_state is not None:
            branch_q = resume_state["q"]
            branch_ids_arr = np.array(branch_ids_arr if branch_ids_arr is not None else np.arange(X), dtype=int)
            if branch_ids_arr.shape[0] != X:
                raise SystemExit(f"[PIMC-FPS] Saved branch_ids length {branch_ids_arr.shape[0]} != X={X}")
            print(f"[PIMC-FPS] Resuming {X} PIMC branches from saved state.")
            for b_idx in range(X):
                branch_configs.append({
                    "q": branch_q[b_idx],
                    "seed_offset": int(branch_ids_arr[b_idx]),
                    "desc": f"resume branch_id={branch_ids_arr[b_idx]}"
                })
        else:
            fps_start_val = fps_start
            try:
                if isinstance(fps_start, str) and fps_start.isdigit():
                    fps_start_val = int(fps_start)
            except:
                pass

            fps_idx = farthest_point_sampling(Qflat, X, rng=np.random.RandomState(base_seed), start=fps_start_val)
            branch_ids_arr = np.array(fps_idx, dtype=int)
            print(f"[PIMC-FPS] Selected FPS frame indices: {fps_idx}")
            for idx in fps_idx:
                branch_configs.append({
                    "q": Qflat[idx],
                    "seed_offset": int(idx),
                    "desc": f"seed_frame={idx}"
                })
        if len(branch_configs) != X:
            raise SystemExit(f"[PIMC-FPS] Expected {X} branch configs but built {len(branch_configs)}")

        # Distribution of samples per subrun
        base = N_total // X
        rem  = N_total % X
        Ns = [base + (1 if i < rem else 0) for i in range(X)]
        print(f"[PIMC-FPS] Sub-run sizes: {Ns}  (sum={sum(Ns)})")

        # Accumulators
        all_q, all_p = [], []
        natoms = None; ndim = None

        def _auto_tune_step(value, rate):
            if rate is None:
                return value, False
            new_val = value
            if rate > accept_target_hi:
                new_val = min(auto_step_max, value * auto_scale_up)
            elif rate < accept_target_lo:
                new_val = max(auto_step_min, value * auto_scale_down)
            return new_val, not math.isclose(new_val, value, rel_tol=1e-9, abs_tol=1e-12)

        # Run each sub-pool
        for i, (cfg, Ni) in enumerate(zip(branch_configs, Ns), 1):
            desc = cfg["desc"]
            seed_offset = cfg["seed_offset"]
            print(f"\n[PIMC-FPS] Sub-run {i}/{X}: {desc} -> N={Ni}")

            branch_seed = get_run_seed(run_id + i * 997 + seed_offset, base_seed)
            rng = np.random.RandomState(branch_seed)
            q0 = np.array(cfg["q"], copy=True)
            if "--init-centroid" in args:
                # Use centroid both at init and afterwards
                centroid = q0.mean(axis=0) if q0.ndim == 2 else q0
                system = System(
                    geometry_file=None,
                    n_beads=n_beads,
                    rng=rng,
                    temperature=T_sim,
                    centroid_q=centroid,
                    symbols=symbols
                )
            else:
                # Default: initialize with centroid, then overwrite with full ring
                centroid = q0.mean(axis=0) if q0.ndim == 2 else q0.mean(axis=0)
                system = System(
                    geometry_file=None,
                    n_beads=n_beads,
                    rng=rng,
                    temperature=T_sim,
                    centroid_q=centroid,
                    symbols=symbols
                )
                # overwrite with full bead configuration, expanding centroid-only frames if needed
                try:
                    q_full = _ensure_frame_nbeads(q0, n_beads, label=desc)
                except ValueError as exc:
                    raise SystemExit(f"[PIMC-FPS] {exc}") from exc
                system.set_positions(q_full)

            if nm_noise and nm_noise > 0:
                did = perturb_ring_polymer_modes(system, nm_noise, max_mode=nm_max_mode)
                if did:
                    print(f"[PIMC-FPS] Injected NM noise σ={nm_noise} (max_mode={nm_max_mode})")

            if natoms is None:
                natoms = system.n_atoms
                ndim   = system.n_dim
            #print(system.positions)
            thermostat = Thermostat(T_sim, tau, thermostat_type, rng)
            sim = Simulation(system, T_sim, timestep, force_field, thermostat,
                            prefix=f"{prefix}_fps{i}",
                            stride_out=0, stride_pos=0, stride_vel=0, stride_log=0)

            # Choose sampling method
            if use_enhanced:
                qs, ps = sim.run_PIMC_enhanced(
                    Ni,
                    out_npz=None,
                    out_text=out_text,
                    gen_velocities=gen_vel,
                    step_size_single=curr_step_single,
                    step_size_disp=curr_step_disp,
                    p_displacement=curr_p_disp,
                    p_single=curr_p_single,
                    p_bisection=curr_p_bis,
                    enable_bisection=enable_bisection)
            elif use_adaptive:
                qs, ps = sim.run_PIMC_adaptive(
                    Ni, step_size=curr_plain_step, out_npz=None, out_text=out_text, gen_velocities=gen_vel)
            else:
                qs, ps = sim.run_PIMC(
                    Ni, step_size=curr_plain_step, out_npz=None, out_text=out_text, gen_velocities=gen_vel)

            stats = getattr(sim, "last_acceptance_stats", None)
            if stats:
                rate = stats.get("acceptance", None)
                if rate is not None:
                    print(f"[PIMC-FPS] Sub-run {i} acceptance={rate:.3f} "
                          f"({stats.get('accepted','?')}/{stats.get('attempted','?')})")
                if stats.get("mode") == "pimc_enhanced":
                    print(f"             move rates single={stats.get('rate_single',0):.3f}, "
                          f"disp={stats.get('rate_disp',0):.3f}, bis={stats.get('rate_bisection',0):.3f}")
                if carry_steps:
                    if stats.get("mode") == "pimc_enhanced":
                        curr_step_single = stats.get("step_single", curr_step_single)
                        curr_step_disp = stats.get("step_disp", curr_step_disp)
                    elif stats.get("mode") == "pimc_single":
                        curr_plain_step = stats.get("step_size", curr_plain_step)
                if auto_adjust and rate is not None:
                    if stats.get("mode") == "pimc_enhanced":
                        tuned = False
                        new_single, changed_single = _auto_tune_step(curr_step_single, rate)
                        new_disp, changed_disp = _auto_tune_step(curr_step_disp, rate)
                        if changed_single:
                            curr_step_single = new_single
                            tuned = True
                        if changed_disp:
                            curr_step_disp = new_disp
                            tuned = True
                        if tuned:
                            print(f"             auto-tuned enhanced steps single={curr_step_single:.4f}, "
                                  f"disp={curr_step_disp:.4f}")
                    elif stats.get("mode") == "pimc_single":
                        new_plain, changed_plain = _auto_tune_step(curr_plain_step, rate)
                        if changed_plain:
                            curr_plain_step = new_plain
                            print(f"             auto-tuned step_size={curr_plain_step:.4f}")

            all_q.append(qs)
            all_p.append(ps)

        # Save merged pool
        Q_all = np.concatenate(all_q, axis=0)
        P_all = np.concatenate(all_p, axis=0)
        # --- Safety checks ---
        print(f"[CHECK] Q_all.shape={Q_all.shape}, P_all.shape={P_all.shape}")
        expected_q_shape = (N_total, n_beads, natoms * ndim)
        expected_p_shape = (N_total, n_beads, natoms * ndim)

        if Q_all.shape != expected_q_shape:
            print(f"[WARN] Unexpected Q_all shape. Expected {expected_q_shape}, got {Q_all.shape}")
        if P_all.shape != expected_p_shape:
            print(f"[WARN] Unexpected P_all shape. Expected {expected_p_shape}, got {P_all.shape}")

        # --- Save merged pool ---
        os.makedirs(os.path.dirname(out_npz_all), exist_ok=True)
        np.savez(out_npz_all,
                q=Q_all,
                p=P_all,
                natoms=natoms,
                ndim=ndim,
                nbeads=n_beads,
                symbols=np.array(symbols),
                meta=f"candidate pool (FPS X={X}, total N={N_total}, dir={direction})")

        print(f"\n[PIMC-FPS] Saved merged pool -> {out_npz_all}")

        # --- Coverage diagnostics ---
        direction_label = direction.upper()
        pool_flat = _flatten_samples(Q_all)
        seed_arr = data["qF"] if direction == "forward" else data["qR"]
        try:
            seed_flat = _flatten_samples(seed_arr)
        except Exception:
            seed_flat = np.zeros((0, pool_flat.shape[1]))
        nn_stats = _nn_to_seed_stats(pool_flat, seed_flat)
        intra_stats = _intra_pool_stats(pool_flat)
        if nn_stats:
            print(f"[COVERAGE:{direction_label}] Pool→seed NN stats: {_format_stats(nn_stats)}")
        if intra_stats:
            print(f"[COVERAGE:{direction_label}] Intra-pool NN stats: {_format_stats(intra_stats)}")

        if state_file:
            if not all_q:
                print(f"[PIMC-FPS] No branch data collected; skipping state save for {state_file}")
            else:
                last_q = np.stack([arr[-1] for arr in all_q], axis=0)
                last_p = np.stack([arr[-1] for arr in all_p], axis=0)
                state_dir = os.path.dirname(state_file)
                if state_dir:
                    os.makedirs(state_dir, exist_ok=True)
                ids_to_store = np.array(branch_ids_arr if branch_ids_arr is not None else np.arange(X), dtype=int)
                np.savez(state_file,
                        q_last=last_q,
                        p_last=last_p,
                        natoms=natoms,
                        ndim=ndim,
                        nbeads=n_beads,
                        branch_ids=ids_to_store,
                        direction=direction,
                        symbols=np.array(symbols))
                print(f"[PIMC-FPS] Saved branch states -> {state_file}")


    # --- GPR TRAIN MODE ---

    if "--gpr-train" in args:
        if not _HAS_GPYTORCH:
            raise SystemExit(f"gpytorch required for GPR mode. "
                             f"Import error: {_GPYTORCH_IMPORT_ERROR}")

        train_npz = args[args.index("--gpr-train")+1]
        model_dir = args[args.index("--gpr-model")+1]

        feature_mode = "centroid"
        if "--feature-mode" in args:
            feature_mode = args[args.index("--feature-mode")+1]

        ard = "--ard" in args
        seed = 123
        if "--seed" in args:
            seed = int(args[args.index("--seed")+1])

        n_iter = 200
        if "--n-iter" in args:
            n_iter = int(args[args.index("--n-iter")+1])

        lr = 0.01
        if "--lr" in args:
            lr = float(args[args.index("--lr")+1])

        noise_min = 1e-6
        if "--noise-min" in args:
            noise_min = float(args[args.index("--noise-min")+1])

        scale_work = "--scale-work" in args
        make_plots = "--gpr-plots" in args
        grouped = "--grouped" in args
        kernel = "rbf"
        if "--kernel" in args:
            kernel = args[args.index("--kernel")+1]
        jitter = 1e-3
        if "--jitter" in args:
            jitter = float(args[args.index("--jitter")+1])
        cg_max_iter = 2000
        if "--cg-max-iter" in args:
            cg_max_iter = int(args[args.index("--cg-max-iter")+1])
        precond_size = 200
        if "--precond-size" in args:
            precond_size = int(args[args.index("--precond-size")+1])

        # Load training data
        data = _load_train_npz(train_npz)

        # Print all hyperparameters
        print("[GPR] Training with gpytorch")
        print(f"        feature_mode = {feature_mode}")
        print(f"        ard          = {ard}")
        print(f"        seed         = {seed}")
        print(f"        n_iter       = {n_iter}")
        print(f"        lr           = {lr}")
        print(f"        noise_min    = {noise_min}")
        print(f"        scale_work   = {scale_work}")
        print(f"        make_plots   = {make_plots}")
        print(f"        jitter       = {jitter}")
        print(f"        cg_max_iter  = {cg_max_iter}")
        print(f"        precond_size = {precond_size}")
        print(f"[GPR] Using {kernel.upper()} kernel "
              f"({'grouped' if grouped else 'ungrouped'})")

        model = WorkGPRTorch(feature_mode=feature_mode,
                             ard=ard,
                             grouped=grouped,
                             kernel=kernel,
                             seed=seed,
                             device="cpu",
                             lr=lr,
                             n_iter=n_iter,
                             noise_min=noise_min,
                             scale_work=scale_work,
                             make_plots=make_plots,
                             jitter=jitter,
                             cg_max_iters=cg_max_iter,
                             precond_size=precond_size)

        # Train forward
        print("[GPR] Training forward GP...")

        fwd_stats = model.fit_forward(data["qF"], data["pF"], data["wF"],
                                      round_num=int(os.path.basename(model_dir).split("_")[-1])
                                      if "round_" in model_dir else None)

        # Train reverse
        print("[GPR] Training reverse GP...")
        rev_stats = model.fit_reverse(data["qR"], data["pR"], data["wR"],
                                      round_num=int(os.path.basename(model_dir).split("_")[-1])
                                      if "round_" in model_dir else None)


        # --- Coverage check ---
        bad_model = False
        for lbl, stats in [("Forward", fwd_stats), ("Reverse", rev_stats)]:
            if stats is None or "coverage" not in stats:
                continue
            cov = stats["coverage"]
            if cov < 0.9:
                print(f"[ERROR] {lbl} GP coverage={cov:.2f} (<0.90). "
                      f"Model will NOT be saved. Reusing previous valid model.")
                bad_model = True
            elif cov < 0.95:
                print(f"[WARN] {lbl} GP coverage={cov:.2f} (<0.95). "
                      f"Consider tuning hyperparameters.")


        os.makedirs(model_dir, exist_ok=True)

        # Save only if coverage is acceptable

        if not bad_model:
            print(f"[GPR] Saving model to: {model_dir}")
            model.save(model_dir)
            np.savez(os.path.join(model_dir, "train_meta.npz"),
                     nF=len(data["qF"]), nR=len(data["qR"]))
            used_round = int(os.path.basename(model_dir).split("_")[-1])
        else:
            print("[GPR] Skipped saving due to low coverage.")
            current_round = int(os.path.basename(model_dir).split("_")[-1])
            prev_round = current_round - 1
            prev_dir = os.path.join(os.path.dirname(model_dir), f"round_{prev_round}")
            if not os.path.isdir(prev_dir):
                raise SystemExit(f"[FATAL] No previous good model found at {prev_dir}")
            shutil.copytree(prev_dir, model_dir, dirs_exist_ok=True)
            print(f"[GPR] Using last good model from round {prev_round} for training round {current_round}")
            used_round = prev_round

        # --- Summary: dataset sizes ---
        # bar_seed
        bar_seed_file = train_npz.replace("gpr_seed", "bar_seed")
        nbar = "unknown"
        if os.path.exists(bar_seed_file):
            dbar = np.load(bar_seed_file, allow_pickle=True)
            nbar = len(dbar["qF"]) + len(dbar["qR"])

        # gpr_seed
        ngpr = len(data["qF"]) + len(data["qR"])

        # model dataset
        train_meta_file = os.path.join(model_dir, "train_meta.npz")
        if os.path.exists(train_meta_file):
            dmeta = np.load(train_meta_file)
            nmodel = int(dmeta["nF"]) + int(dmeta["nR"])
        else:
            nmodel = "unknown"

        with open(os.path.join(model_dir, "feature_mode.txt"), "w") as f:
            f.write(feature_mode)

        print(f"[SUMMARY] bar_seed={nbar}, gpr_seed={ngpr}, model_trained_on={nmodel} (from round {used_round})")

        return

    _handle_remaining_cli(args)


# --- GPR PREDICT/RANK MODE ---
# --- Helper for uncertainty selection ---

def _select_by_uncertainty(std, topk=None, threshold=None):
    std = np.asarray(std)
    n = len(std)

    if threshold is not None:
        # Split by threshold
        hi_idx = np.where(std > threshold)[0]
        lo_idx = np.where(std <= threshold)[0]

        # Enforce minimum topk if requested
        if topk is not None and len(hi_idx) < topk:
            # Sort all indices by descending uncertainty
            idx_sorted = np.argsort(-std)
            hi_idx = idx_sorted[:min(topk, n)]
            lo_idx = np.setdiff1d(np.arange(n), hi_idx)

    else:
        # Pure top-k mode
        idx_sorted = np.argsort(-std)  # descending
        if topk is None:
            topk = 100
        hi_idx = idx_sorted[:min(topk, n)]
        lo_idx = np.setdiff1d(np.arange(n), hi_idx)

    return hi_idx, lo_idx


def _subsample_low_indices(idx_all, features, max_keep=None,
                           method="uniform", seed=1234):
    idx_all = np.asarray(idx_all)
    if max_keep is None or max_keep <= 0 or len(idx_all) <= max_keep:
        return idx_all
    rng = np.random.default_rng(seed)
    if method == "uniform":
        chosen = np.sort(rng.choice(len(idx_all), size=max_keep, replace=False))
        return idx_all[chosen]
    elif method == "fps":
        pts = features[idx_all]
        fps_rng = np.random.RandomState(seed)
        fps_idx = farthest_point_sampling(pts, max_keep, rng=fps_rng, start="auto")
        fps_idx = np.sort(fps_idx)
        return idx_all[fps_idx]
    else:
        raise ValueError(f"[GPR] Unknown subsample method '{method}'")

def _scaled_features(block, q, p):
    q_arr = q if q.ndim == 2 else q.reshape(len(q), -1)
    p_arr = p if p.ndim == 2 else p.reshape(len(p), -1)
    q_feat = block["scaler_q"].transform(q_arr)
    p_feat = block["scaler_p"].transform(p_arr)
    return np.concatenate([q_feat, p_feat], axis=1)


def _handle_remaining_cli(args):
    #GRP predict both directions

    if "--gpr-predict-both" in args:
        if not _HAS_GPYTORCH:
            raise SystemExit("gpytorch required for GPR mode.")

        candF_npz = args[args.index("--cands-forward") + 1]
        candR_npz = args[args.index("--cands-reverse") + 1]
        model_dir = args[args.index("--gpr-model") + 1]

        outF_hi = args[args.index("--select-out-forward") + 1]
        outF_lo = args[args.index("--select-keep-forward") + 1] if "--select-keep-forward" in args else "idx_lo_forward.npy"
        outR_hi = args[args.index("--select-out-reverse") + 1]
        outR_lo = args[args.index("--select-keep-reverse") + 1] if "--select-keep-reverse" in args else "idx_lo_reverse.npy"

        topk = int(args[args.index("--topk") + 1]) if "--topk" in args else None
        thresh = float(args[args.index("--unc-threshold") + 1]) if "--unc-threshold" in args else None
        thresh_rel = float(args[args.index("--unc-threshold-rel") + 1]) if "--unc-threshold-rel" in args else None
        lo_subset_global = int(args[args.index("--lo-subset")+1]) if "--lo-subset" in args else None
        lo_subset_forward = int(args[args.index("--lo-subset-forward")+1]) if "--lo-subset-forward" in args else None
        lo_subset_reverse = int(args[args.index("--lo-subset-reverse")+1]) if "--lo-subset-reverse" in args else None
        lo_subset_method = args[args.index("--lo-subset-method")+1] if "--lo-subset-method" in args else "uniform"
        lo_subset_seed = int(args[args.index("--lo-subset-seed")+1]) if "--lo-subset-seed" in args else 1234

        # --- Load model and its feature_mode ---
        model = WorkGPRTorch.load(model_dir)
        fm_file = os.path.join(model_dir, "feature_mode.txt")
        if os.path.exists(fm_file):
            with open(fm_file) as f:
                model.feature_mode = f.read().strip()
        else:
            model.feature_mode = "centroid"
        print(f"[GPR] Using feature_mode from model: {model.feature_mode}")

        # === Forward ===
        candF = _load_cand_npz(candF_npz)
        qF, pF = preprocess_features(candF["q"], candF["p"], model, label="FWD")

        if thresh_rel is not None:
            sigma_w = model.gp_forward["train_w_std"]
            threshF = thresh_rel * sigma_w
            print(f"[GPR:F] Using relative uncertainty threshold: {thresh_rel} × {sigma_w:.4f} = {threshF:.4f}")
        else:
            threshF = thresh

        muF, stdF = model.predict_forward(qF, pF)
        idxF_hi, idxF_lo = _select_by_uncertainty(stdF, topk=topk, threshold=threshF)
        lo_keep = lo_subset_forward if lo_subset_forward is not None else lo_subset_global
        if lo_keep is not None and lo_keep > 0:
            featF = _scaled_features(model.gp_forward, qF, pF)
            before = len(idxF_lo)
            idxF_lo = _subsample_low_indices(idxF_lo, featF, max_keep=lo_keep,
                                             method=lo_subset_method, seed=lo_subset_seed)
            if len(idxF_lo) < before:
                print(f"[GPR:F] Subsampled low-uncertainty pool {before}→{len(idxF_lo)} "
                      f"(method={lo_subset_method}, max_keep={lo_keep})")
        print(f"[GPR:F] Selected {len(idxF_hi)} high-uncertainty and {len(idxF_lo)} low-uncertainty samples.")

        np.save(outF_hi, idxF_hi)
        np.save(outF_lo, idxF_lo)
        np.savez(outF_lo.replace("idx_lo", "predicted_lo").replace(".npy", ".npz"),
                q=candF["q"][idxF_lo], p=candF["p"][idxF_lo], w=muF[idxF_lo])
        np.savez(outF_hi.replace("idx_hi", "predicted_hi").replace(".npy", ".npz"),
                q=candF["q"][idxF_hi], p=candF["p"][idxF_hi], w=muF[idxF_hi], std=stdF[idxF_hi])
        np.savez(outF_lo.replace("idx_lo", "predicted_all").replace(".npy", ".npz"),
                q=candF["q"], p=candF["p"], w=muF, std=stdF)

        # === Reverse ===
        candR = _load_cand_npz(candR_npz)
        qR, pR = preprocess_features(candR["q"], candR["p"], model, label="REV")

        if thresh_rel is not None:
            sigma_w = model.gp_reverse["train_w_std"]
            threshR = thresh_rel * sigma_w
            print(f"[GPR:R] Using relative uncertainty threshold: {thresh_rel} × {sigma_w:.4f} = {threshR:.4f}")
        else:
            threshR = thresh

        muR, stdR = model.predict_reverse(qR, pR)
        idxR_hi, idxR_lo = _select_by_uncertainty(stdR, topk=topk, threshold=threshR)
        lo_keep_R = lo_subset_reverse if lo_subset_reverse is not None else lo_subset_global
        if lo_keep_R is not None and lo_keep_R > 0:
            featR = _scaled_features(model.gp_reverse, qR, pR)
            before = len(idxR_lo)
            idxR_lo = _subsample_low_indices(idxR_lo, featR, max_keep=lo_keep_R,
                                             method=lo_subset_method, seed=lo_subset_seed + 1)
            if len(idxR_lo) < before:
                print(f"[GPR:R] Subsampled low-uncertainty pool {before}→{len(idxR_lo)} "
                      f"(method={lo_subset_method}, max_keep={lo_keep_R})")
        print(f"[GPR:R] Selected {len(idxR_hi)} high-uncertainty and {len(idxR_lo)} low-uncertainty samples.")

        np.save(outR_hi, idxR_hi)
        np.save(outR_lo, idxR_lo)
        np.savez(outR_lo.replace("idx_lo", "predicted_lo").replace(".npy", ".npz"),
                q=candR["q"][idxR_lo], p=candR["p"][idxR_lo], w=muR[idxR_lo])
        np.savez(outR_hi.replace("idx_hi", "predicted_hi").replace(".npy", ".npz"),
                q=candR["q"][idxR_hi], p=candR["p"][idxR_hi], w=muR[idxR_hi], std=stdR[idxR_hi])
        np.savez(outR_lo.replace("idx_lo", "predicted_all").replace(".npy", ".npz"),
                q=candR["q"], p=candR["p"], w=muR, std=stdR)
        return



    if "--run-spimd-from-pool" in args:
        pool_npz = args[args.index("--pool-npz") + 1]
        idx_file = args[args.index("--select-indices") + 1]
        direction = args[args.index("--direction") + 1].lower()
        n_steps = int(args[args.index("--nsteps") + 1])
        input_file = args[args.index("--input-file") + 1] if "--input-file" in args else "input.txt"
        out_npz = args[args.index("--out") + 1] if "--out" in args else "work_from_spimd.npz"

        assert direction in {"forward", "reverse"}, "direction must be forward or reverse"

        # Load candidates
        print(f"[SPIMD] Loading pool from: {pool_npz}")
        pool = np.load(pool_npz, allow_pickle=True)
        q_pool = pool["q"]   # shape: (N, nbeads, ndim*natoms) or (N, ndim*natoms)
        p_pool = pool["p"]
        symbols = pool["symbols"].tolist()
        natoms = int(pool["natoms"])
        ndim = int(pool["ndim"])
        nbeads = int(pool["nbeads"])
        target_nbeads = nbeads
        if "--nbeads" in args:
            target_nbeads = int(args[args.index("--nbeads") + 1])
            if target_nbeads != nbeads:
                print(f"[SPIMD] Overriding pool nbeads={nbeads} → target {target_nbeads}")
        try:
            q_pool = _ensure_pool_nbeads(q_pool, target_nbeads, label="q_pool")
            p_pool = _ensure_pool_nbeads(p_pool, target_nbeads, label="p_pool")
        except ValueError as exc:
            raise SystemExit(f"[SPIMD] {exc}") from exc
        nbeads = target_nbeads
        print(f"[SPIMD] Pool contains {len(q_pool)} samples with {nbeads} beads, {natoms} atoms, {ndim} dimensions.")
        print(f"[SPIMD] Loading selected indices from: {idx_file}")
        selected_idx = np.load(idx_file)

        # Prepare input system
        inp = read_input_file(input_file)
        master_seed = inp.get("seed", 420)
        run_id = inp.get("run_id", 0)
        system_temperature = float(inp["temperature"])
        simulation_temperature = system_temperature * nbeads
        timestep = inp["timestep"]
        prefix = inp.get("prefix", "spimd_pool")
        tau = inp.get("tau", 1)
        thermostat_type = inp.get("thermostat", None)

        # Initialize model and thermostat (outside loop)
        force_field = Force_Field(inp["forces"], direction=direction)

        # Output work values
        works = []

        for count, i in enumerate(selected_idx):

            #print('la reverga',p_pool[i])
            rng = np.random.RandomState(get_run_seed(run_id + count, master_seed))

            if "--init-centroid" in args:
                # Use centroid both at init and afterwards
                centroid = q_pool[i].mean(axis=0) if q_pool[i].ndim == 2 else q_pool[i]
                system = System(
                    geometry_file=None,
                    n_beads=nbeads,
                    rng=rng,
                    temperature=simulation_temperature,
                    centroid_q=centroid,
                    symbols=symbols
                )
            else:
                # Default: initialize with centroid, then overwrite with full ring
                centroid = q_pool[i].mean(axis=0) if q_pool[i].ndim == 2 else q_pool[i].mean(axis=0)
                system = System(
                    geometry_file=None,
                    n_beads=nbeads,
                    rng=rng,
                    temperature=simulation_temperature,
                    centroid_q=centroid,
                    symbols=symbols
                )
                try:
                    q_full = _ensure_frame_nbeads(q_pool[i], nbeads, label=f"q_pool[{i}]")
                except ValueError as exc:
                    raise SystemExit(f"[SPIMD] {exc}") from exc
                system.set_positions(q_full)

            # Overwrite with provided momenta
            try:
                p_full = _ensure_frame_nbeads(p_pool[i], nbeads, label=f"p_pool[{i}]")
            except ValueError as exc:
                raise SystemExit(f"[SPIMD] {exc}") from exc
            system.velocities = p_full

            thermostat = Thermostat(simulation_temperature, tau, thermostat_type, rng)
            simulation = Simulation(
                system, simulation_temperature, timestep, force_field, thermostat,
                prefix=f"{prefix}_{direction}_{i}",stride_out=0, stride_pos=0, stride_vel=0, stride_log=1000000
            )
            print(f"[SPIMD] Running index {i} ({direction})...")
            work = simulation.run_SPIMD(n_steps, compute_work=True)
            works.append(work)
            #print(f"[SPIMD] Work: {work:.4f}")

        # Save work, positions, and momenta
        q_selected = q_pool[selected_idx]
        p_selected = p_pool[selected_idx]

        os.makedirs(os.path.dirname(out_npz), exist_ok=True)

        if direction == "forward":
            np.savez(out_npz,
                    wF=np.array(works),
                    q=q_selected,
                    p=p_selected,
                    natoms=natoms,
                    ndim=ndim,
                    nbeads=nbeads,
                    meta="SPIMD forward run")
        else:
            np.savez(out_npz,
                    wR=np.array(works),
                    q=q_selected,
                    p=p_selected,
                    natoms=natoms,
                    ndim=ndim,
                    nbeads=nbeads,
                    meta="SPIMD reverse run")

        print(f"[SPIMD] Saved {len(works)} work values (+q/p) to {out_npz}")
        return

    if "--merge-data" in args:

        seed_path = args[args.index("--seed-old")+1]
        out_path  = args[args.index("--out")+1]

        # Optional: reproducibility for FPS
        fps_seed = 123
        if "--fps-seed" in args:
            fps_seed = int(args[args.index("--fps-seed")+1])

        # Optional: starting rule for FPS
        fps_start = "auto"
        if "--fps-start" in args:
            fps_start = args[args.index("--fps-start")+1]
            try:
                fps_start = int(fps_start)
            except ValueError:
                pass

        rng = np.random.RandomState(fps_seed)

        flood_threshold = float(args[args.index("--flood-threshold")+1]) if "--flood-threshold" in args else 5.0
        train_select = int(args[args.index("--train-select")+1]) if "--train-select" in args else 100
        if train_select < 0:
            raise SystemExit("[MERGE] --train-select must be non-negative")

        add_forward, add_reverse = [], []
        if "--add-forward" in args:
            i = args.index("--add-forward")+1
            while i < len(args) and not args[i].startswith("--"):
                add_forward.append(args[i]); i += 1
        if "--add-reverse" in args:
            i = args.index("--add-reverse")+1
            while i < len(args) and not args[i].startswith("--"):
                add_reverse.append(args[i]); i += 1

        print(f"[MERGE] Base seed: {seed_path}")
        seed = np.load(seed_path, allow_pickle=True)
        qF, pF, wF = seed["qF"], seed["pF"], seed["wF"]
        qR, pR, wR = seed["qR"], seed["pR"], seed["wR"]
        symbols = seed["symbols"].tolist()
        natoms, ndim, nbeads = int(seed["natoms"]), int(seed["ndim"]), int(seed["nbeads"])
        nF_before, nR_before = len(qF), len(qR)

        # Collect sanitized SPIMD + prediction data
        qF_sanitized, pF_sanitized, wF_sanitized = [], [], []
        qR_sanitized, pR_sanitized, wR_sanitized = [], [], []
        pred_forward = spimd_forward = 0
        pred_reverse = spimd_reverse = 0

        # ---- Forward ----
        for path in add_forward:
            d = np.load(path, allow_pickle=True)
            try:
                qF_add = _ensure_pool_nbeads(d["q"], nbeads, label=f"{path}:q")
                pF_add = _ensure_pool_nbeads(d["p"], nbeads, label=f"{path}:p")
            except ValueError as exc:
                raise SystemExit(f"[MERGE] {exc}") from exc
            if "wF" in d:
                wF_add = d["wF"]
                spimd_forward += len(wF_add)
            elif "w" in d:
                wF_add = d["w"]
                pred_forward += len(wF_add)
            else:
                wF_add = np.zeros(len(qF_add))
                print(f"[WARN] {path} has no work key → filling zeros")
            qF_sanitized.append(qF_add); pF_sanitized.append(pF_add); wF_sanitized.append(wF_add)
            print(f"[MERGE] Added FORWARD {len(qF_add)} from {path}")

        # ---- Reverse ----
        for path in add_reverse:
            d = np.load(path, allow_pickle=True)
            try:
                qR_add = _ensure_pool_nbeads(d["q"], nbeads, label=f"{path}:q")
                pR_add = _ensure_pool_nbeads(d["p"], nbeads, label=f"{path}:p")
            except ValueError as exc:
                raise SystemExit(f"[MERGE] {exc}") from exc
            if "wR" in d:
                wR_add = d["wR"]
                spimd_reverse += len(wR_add)
            elif "w" in d:
                wR_add = d["w"]
                pred_reverse += len(wR_add)
            else:
                wR_add = np.zeros(len(qR_add))
                print(f"[WARN] {path} has no work key → filling zeros")
            qR_sanitized.append(qR_add); pR_sanitized.append(pR_add); wR_sanitized.append(wR_add)
            print(f"[MERGE] Added REVERSE {len(qR_add)} from {path}")

        # Concatenate if any additions
        if qF_sanitized:
            qF_sanitized = np.concatenate(qF_sanitized, axis=0)
            pF_sanitized = np.concatenate(pF_sanitized, axis=0)
            wF_sanitized = np.concatenate(wF_sanitized, axis=0)
        else:
            qF_sanitized = np.zeros((0,)+qF.shape[1:])
            pF_sanitized = np.zeros((0,)+pF.shape[1:])
            wF_sanitized = np.zeros((0,))
        if qR_sanitized:
            qR_sanitized = np.concatenate(qR_sanitized, axis=0)
            pR_sanitized = np.concatenate(pR_sanitized, axis=0)
            wR_sanitized = np.concatenate(wR_sanitized, axis=0)
        else:
            qR_sanitized = np.zeros((0,)+qR.shape[1:])
            pR_sanitized = np.zeros((0,)+pR.shape[1:])
            wR_sanitized = np.zeros((0,))

        # === Option 2 ===
        qF_free = np.concatenate([qF, qF_sanitized], axis=0)
        pF_free = np.concatenate([pF, pF_sanitized], axis=0)
        wF_free = np.concatenate([wF, wF_sanitized], axis=0)
        qR_free = np.concatenate([qR, qR_sanitized], axis=0)
        pR_free = np.concatenate([pR, pR_sanitized], axis=0)
        wR_free = np.concatenate([wR, wR_sanitized], axis=0)

        # Training dataset: farthest-point sample sanitized SPIMD points
        n_select = min(train_select, len(qF_sanitized))
        if n_select > 0:
            idxF = farthest_point_sampling(qF_sanitized.reshape(len(qF_sanitized), -1),
                                        n_select, rng=rng, start=fps_start)
            qF_train = np.concatenate([qF, qF_sanitized[idxF]], axis=0)
            pF_train = np.concatenate([pF, pF_sanitized[idxF]], axis=0)
            wF_train = np.concatenate([wF, wF_sanitized[idxF]], axis=0)
        else:
            qF_train, pF_train, wF_train = qF, pF, wF
        n_select = min(train_select, len(qR_sanitized))
        if n_select > 0:
            idxR = farthest_point_sampling(qR_sanitized.reshape(len(qR_sanitized), -1),
                                        n_select, rng=rng, start=fps_start)
            qR_train = np.concatenate([qR, qR_sanitized[idxR]], axis=0)
            pR_train = np.concatenate([pR, pR_sanitized[idxR]], axis=0)
            wR_train = np.concatenate([wR, wR_sanitized[idxR]], axis=0)
        else:
            qR_train, pR_train, wR_train = qR, pR, wR

        # === Save correctly depending on directory ===
        if "bar_seed" in out_path:
            np.savez(out_path,
                    qF=qF_free, pF=pF_free, wF=wF_free,
                    qR=qR_free, pR=pR_free, wR=wR_free,
                    symbols=symbols, natoms=natoms, ndim=ndim, nbeads=nbeads)
            print(f"[MERGE] Saved BAR dataset → {out_path}")
        elif "gpr_seed" in out_path:
            np.savez(out_path,
                    qF=qF_train, pF=pF_train, wF=wF_train,
                    qR=qR_train, pR=pR_train, wR=wR_train,
                    symbols=symbols, natoms=natoms, ndim=ndim, nbeads=nbeads)
            print(f"[MERGE] Saved GPR dataset → {out_path}")
        else:
            raise RuntimeError(f"[MERGE] Cannot determine dataset type from path: {out_path}")

        print(f"[SUMMARY] Forward: base {nF_before}, free={len(qF_free)}, train={len(qF_train)}")
        print(f"[SUMMARY] Reverse: base {nR_before}, free={len(qR_free)}, train={len(qR_train)}")

        def _warn_flood(pred_ct, spimd_ct, label):
            if pred_ct == 0:
                return
            if spimd_ct == 0:
                print(f"[WARN] {label}: {pred_ct} predicted samples but no SPIMD data. "
                      f"Crooks/BAR may rely solely on GP outputs.")
                return
            ratio = pred_ct / max(1, spimd_ct)
            if ratio > flood_threshold:
                print(f"[WARN] {label}: predicted/SPIMD ratio={ratio:.2f} "
                      f"({pred_ct}/{spimd_ct}) exceeds threshold {flood_threshold:.2f}.")

        _warn_flood(pred_forward, spimd_forward, "FORWARD merge")
        _warn_flood(pred_reverse, spimd_reverse, "REVERSE merge")

        return

    # --- BAR/SEED (Default) MODE ---
    if "--help" in args or len(args) == 0:
        print("Usage:\n"
            "  # === BAR free energy from simulation directories ===\n"
            "  controller.py <forward_root> <reverse_root> <beta> \\\n"
            "                [--prefix-work NAME] [--out OUT.npz]\n"
            "\n"
            "  # === BAR from existing seed file ===\n"
            "  controller.py --bar-from-seed SEED.npz [--out OUT.npz]\n"
            "\n"
            "  # === Run SPIMD simulations from selected pool indices ===\n"
            "  controller.py --run-spimd-from-pool \\\n"
            "                --pool-npz POOL.npz --select-indices INDICES.npy \\\n"
            "                --direction forward|reverse --nsteps N \\\n"
            "                --input-file input.txt --out OUT.npz\n"
            "\n"
            "  # === Merge predicted_lo and SPIMD data into new seed ===\n"
            "  controller.py --merge-data \\\n"
            "                --seed-old SEED_IN.npz --out SEED_OUT.npz \\\n"
            "                --add-forward FILES... --add-reverse FILES... \\\n"
            "                [--flood-threshold R] [--train-select N]\n"
            "\n"
            "  # === GPR training ===\n"
            "  controller.py --gpr-train TRAIN.npz --gpr-model MODEL_DIR \\\n"
            "                [--feature-mode centroid|raw|nmodes:*] [--restarts N] [--ard] [--seed S] \\\n"
            "                [--jitter EPS] [--cg-max-iter N] [--precond-size M]\n"
            "\n"
            "  # === GPR prediction (uncertainty ranking) ===\n"
            "  controller.py --gpr-predict-forward CANDS.npz --gpr-model MODEL_DIR \\\n"
            "                --select-out IDX.npy [--topk K] [--outdir DIR] \\\n"
            "                [--lo-subset N] [--lo-subset-method uniform|fps] [--lo-subset-seed S]\n"
            "  controller.py --gpr-predict-reverse CANDS.npz --gpr-model MODEL_DIR \\\n"
            "                --select-out IDX.npy [--topk K] [--outdir DIR] \\\n"
            "                [--lo-subset N] [--lo-subset-method uniform|fps] [--lo-subset-seed S]\n"
            "  controller.py --gpr-predict-both --cands-forward F.npz --cands-reverse R.npz \\\n"
            "                --gpr-model MODEL_DIR \\\n"
            "                --select-out-forward IDXF.npy --select-out-reverse IDXR.npy \\\n"
            "                [--topk K] [--outdir DIR] \\\n"
            "                [--lo-subset N | --lo-subset-forward NF --lo-subset-reverse NR] \\\n"
            "                [--lo-subset-method uniform|fps] [--lo-subset-seed S]\n"
            "\n"
            "  # === GPR prediction (uncertainty ranking) ===\n"
            "  controller.py --gpr-predict-forward CANDS.npz --gpr-model MODEL_DIR \\\n"
            "                --select-out IDX.npy [--topk K] [--outdir DIR]\n"
            "  controller.py --gpr-predict-reverse CANDS.npz --gpr-model MODEL_DIR \\\n"
            "                --select-out IDX.npy [--topk K] [--outdir DIR]\n"
            "  controller.py --gpr-predict-both --cands-forward F.npz --cands-reverse R.npz \\\n"
            "                --gpr-model MODEL_DIR \\\n"
            "                --select-out-forward IDXF.npy --select-out-reverse IDXR.npy \\\n"
            "                [--topk K] [--outdir DIR] \\\n"
            "                [--lo-subset N | --lo-subset-forward NF --lo-subset-reverse NR] \\\n"
            "                [--lo-subset-method uniform|fps] [--lo-subset-seed S]\n"
            "\n"
            "  # === Seed update with lowest-uncertainty predicted works ===\n"
            "  controller.py --seed-update-from-pool \\\n"
            "                --seed-in SEED.npz --seed-out UPDATED.npz \\\n"
            "                --gpr-model MODEL_DIR \\\n"
            "                [--cands-forward F.npz] [--cands-reverse R.npz] \\\n"
            "                [--k-low N]\n"
            "\n"
            "  # === Generate PIMC pool ===\n"
            "  controller.py --pimc-pool OUTFILE.npz --N N --nbeads NB \\\n"
            "                --seed-npz SEED.npz --direction forward|reverse \\\n"
            "                [--no-text|--out-text]\n"
            "\n"
            "  controller.py --pimc-fps-pool OUTFILE.npz --N N --X X --nbeads NB \\\n"
            "                --seed-npz SEED.npz --direction forward|reverse \\\n"
            "                [--state-file STATE.npz] [...]\n")
        sys.exit(0)


    # === BAR from raw simulation outputs ===
    if len(args) >= 3 and not any("--" in a for a in args[:3]):
        forward_root, reverse_root, beta = args[0], args[1], float(args[2])
        prefix_work = args[args.index("--prefix-work")+1] if "--prefix-work" in args else "simulation"
        out_npz = args[args.index("--out")+1] if "--out" in args else "data/seed_data.npz"
        use_beads = "--beads" in args
        # Optional: number of data points
        ndata = None
        if "--ndata" in args:
            ndata = int(args[args.index("--ndata")+1])

        filesF = find_work_files(forward_root, prefix_work)
        filesR = find_work_files(reverse_root, prefix_work)

        print(f"[INFO] prefix-work='{prefix_work}'")
        print(f"[INFO] Found work files: F={len(filesF)}, R={len(filesR)}")

        WF, WR = [], []
        QF, PF, QR, PR = [], [], [], []
        natoms, ndim, nbeads = None, None, 1
        symbols = None

        # Forward runs
        for fp in filesF:
            try:
                works = read_all_works(fp)
                run_dir = os.path.dirname(fp)

                if use_beads:
                    bead_qs, bead_ps = [], []
                    bead_idx = 0
                    while True:
                        pos_path = os.path.join(run_dir, f"simulation.pos{bead_idx}.xyz")
                        vel_path = os.path.join(run_dir, f"simulation.vel{bead_idx}.xyz")
                        if not os.path.exists(pos_path) or not os.path.exists(vel_path):
                            break
                        q_frame, nat, nd, syms = read_multi_xyz(pos_path)  # (n_frames, D)
                        p_frame, _, _, _       = read_multi_xyz(vel_path)  # (n_frames, D)
                        if natoms is None:
                            natoms, ndim, symbols = nat, nd, syms
                        bead_qs.append(q_frame)
                        bead_ps.append(p_frame)
                        bead_idx += 1

                    if bead_idx == 0:
                        raise RuntimeError(f"No bead files found in {run_dir}")

                    nbeads = bead_idx
                    q_all = np.stack(bead_qs, axis=1)  # (n_frames, nbeads, D)
                    p_all = np.stack(bead_ps, axis=1)  # (n_frames, nbeads, D)
                else:
                    # centroid-only fallback
                    q_all, nat, nd, syms = read_multi_xyz(os.path.join(run_dir, "simulation.xc.xyz"))
                    p_all, _, _, _       = read_multi_xyz(os.path.join(run_dir, "simulation.vc.xyz"))
                    if natoms is None:
                        natoms, ndim, symbols = nat, nd, syms
                    q_all = q_all.reshape(q_all.shape[0], 1, nat*nd)
                    p_all = p_all.reshape(p_all.shape[0], 1, nat*nd)
                    nbeads = 1

                print(f"[DEBUG] Forward run {run_dir}: q_all.shape={q_all.shape}, p_all.shape={p_all.shape}")

                QF.append(q_all)
                PF.append(p_all)
                WF.extend(works)

            except Exception as e:
                print(f"[WARN] skip F run {fp}")
                traceback.print_exc()

        # Reverse runs
        for rp in filesR:
            try:
                works = read_all_works(rp)
                run_dir = os.path.dirname(rp)

                if use_beads:
                    bead_qs, bead_ps = [], []
                    bead_idx = 0
                    while True:
                        pos_path = os.path.join(run_dir, f"simulation.pos{bead_idx}.xyz")
                        vel_path = os.path.join(run_dir, f"simulation.vel{bead_idx}.xyz")
                        if not os.path.exists(pos_path) or not os.path.exists(vel_path):
                            break
                        q_frame, nat, nd, syms = read_multi_xyz(pos_path)
                        p_frame, _, _, _       = read_multi_xyz(vel_path)
                        if natoms is None:
                            natoms, ndim, symbols = nat, nd, syms
                        bead_qs.append(q_frame)
                        bead_ps.append(p_frame)
                        bead_idx += 1

                    if bead_idx == 0:
                        raise RuntimeError(f"No bead files found in {run_dir}")

                    nbeads = bead_idx
                    q_all = np.stack(bead_qs, axis=1)
                    p_all = np.stack(bead_ps, axis=1)
                else:
                    q_all, nat, nd, syms = read_multi_xyz(os.path.join(run_dir, "simulation.xc.xyz"))
                    p_all, _, _, _       = read_multi_xyz(os.path.join(run_dir, "simulation.vc.xyz"))
                    if natoms is None:
                        natoms, ndim, symbols = nat, nd, syms
                    q_all = q_all.reshape(q_all.shape[0], 1, nat*nd)
                    p_all = p_all.reshape(p_all.shape[0], 1, nat*nd)
                    nbeads = 1

                print(f"[DEBUG] Reverse run {run_dir}: q_all.shape={q_all.shape}, p_all.shape={p_all.shape}")

                QR.append(q_all)
                PR.append(p_all)
                WR.extend(works)

            except Exception as e:
                print(f"[WARN] skip R run {rp}")
                traceback.print_exc()


        # --- Guard against empty data ---
        if not QF or not PF:
            sys.exit(f"[ERROR] No valid FORWARD runs found in {forward_root}. Aborting.")
        if not QR or not PR:
            sys.exit(f"[ERROR] No valid REVERSE runs found in {reverse_root}. Aborting.")

        # Concatenate
        WF, WR = np.array(WF, float), np.array(WR, float)
        qF, pF = np.concatenate(QF, axis=0), np.concatenate(PF, axis=0)
        qR, pR = np.concatenate(QR, axis=0), np.concatenate(PR, axis=0)

        # Restrict number of datapoints if requested (per direction!)
        if ndata is not None:
            nF = min(len(WF), ndata)
            nR = min(len(WR), ndata)
            qF, pF, WF = qF[:nF], pF[:nF], WF[:nF]
            qR, pR, WR = qR[:nR], pR[:nR], WR[:nR]
            print(f"[BAR] Using {nF} forward and {nR} reverse datapoints")

        print(f"[INFO] Totals: F={len(WF)}, R={len(WR)}")
        print(f"[INFO] Shapes qF={qF.shape}, pF={pF.shape}, qR={qR.shape}, pR={pR.shape}")

        # BAR estimate
        df = bar_estimate(WF, WR, beta)
        ci, info = bar_ci(WF, WR, beta, df)
        print(f"[BAR] ΔF = {df:.8f}")
        print(f"[BAR] 95% CI = [{ci[0]:.8f}, {ci[1]:.8f}] width={ci[2]:.8f}")
        print(f"[BAR] Fisher info = {info:.2f}")

        # Save NPZ
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        meta_tag = "seed with bead ICs" if use_beads else "seed with centroid ICs"

        np.savez(out_npz,
                 qF=qF, pF=pF, wF=WF,
                 qR=qR, pR=pR, wR=WR,
                 natoms=natoms, ndim=ndim, nbeads=nbeads,
                 symbols=np.array(symbols),
                 beta=beta,
                 meta=meta_tag)
        print(f"[SAVE] wrote {out_npz}")



    # === BAR from existing SEED ===
    elif "--bar-from-seed" in args:

        if "--beta" not in args:
            print("[ERROR] --beta is required for BAR-from-seed mode")
            sys.exit(1)

        seed_file = args[args.index("--bar-from-seed")+1]
        out_npz   = args[args.index("--out")+1] if "--out" in args else "data/seed_data.npz"
        beta      = float(args[args.index("--beta")+1])

        # Optional: number of data points
        ndata = None
        if "--ndata" in args:
            ndata = int(args[args.index("--ndata")+1])

        print(f"[BAR] Loading seed from: {seed_file}")
        if not os.path.exists(seed_file):
            print(f"[ERROR] Seed file not found: {seed_file}")
            sys.exit(1)

        data = np.load(seed_file, allow_pickle=True)

        # Mandatory arrays
        try:
            qF, pF, wF = data["qF"], data["pF"], data["wF"]
            qR, pR, wR = data["qR"], data["pR"], data["wR"]
        except KeyError as e:
            print(f"[ERROR] Missing expected key in {seed_file}: {e}")
            sys.exit(1)
        wR = wR
        symbols = data["symbols"].tolist() if "symbols" in data else []
        natoms  = int(data["natoms"]) if "natoms" in data else -1
        ndim    = int(data["ndim"]) if "ndim" in data else -1
        nbeads  = int(data["nbeads"]) if "nbeads" in data else -1

        # Restrict number of datapoints if requested
        if ndata is not None:
            qF, pF, wF = qF[:ndata], pF[:ndata], wF[:ndata]
            qR, pR, wR = qR[:ndata], pR[:ndata], wR[:ndata]
            print(f"[BAR] Using first {ndata} datapoints")

        print(f"[INFO] Totals: F={len(wF)}, R={len(wR)}")
        if len(wF) == 0 or len(wR) == 0:
            print("[ERROR] No work data found in seed file → cannot compute BAR")
            sys.exit(1)

        print(f"[INFO] Shapes qF={qF.shape}, pF={pF.shape}, qR={qR.shape}, pR={pR.shape}")
        print(f"[DEBUG] First works: F={wF[:5]} R={wR[:5]}")

        # BAR computation
        try:
            df = bar_estimate(wF, wR, beta)
            ci, info = bar_ci(wF, wR, beta, df)
        except Exception as e:
            print(f"[ERROR] BAR computation failed: {e}")
            sys.exit(1)

        print(f"[BAR] ΔF = {df:.8f}")
        print(f"[BAR] 95% CI = [{ci[0]:.8f}, {ci[1]:.8f}] width={ci[2]:.8f}")
        print(f"[BAR] Fisher info = {info:.2f}")

        np.savez(out_npz,
                qF=qF, pF=pF, wF=wF,
                qR=qR, pR=pR, wR=wR,
                natoms=natoms, ndim=ndim, nbeads=nbeads,
                symbols=np.array(symbols),
                beta=beta,
                meta="seed with centroid ICs (from round merge)")
        print(f"[SAVE] wrote {out_npz}")

    # === Crooks from existing SEED ===
    elif "--crooks-from-seed" in args:
        seed_file = args[args.index("--crooks-from-seed")+1]
        nboot = int(args[args.index("--bootstrap")+1]) if "--bootstrap" in args else 200

        print(f"[Crooks] Loading seed from: {seed_file}")
        if not os.path.exists(seed_file):
            sys.exit(f"[ERROR] Seed file not found: {seed_file}")

        data = np.load(seed_file, allow_pickle=True)
        try:
            wF, wR = np.asarray(data["wF"], float), np.asarray(data["wR"], float)
        except KeyError as e:
            sys.exit(f"[ERROR] Missing expected key in {seed_file}: {e}")

        if len(wF) == 0 or len(wR) == 0:
            sys.exit("[ERROR] No work data in seed file → cannot compute Crooks")

        # --- Bootstrap Crooks ---
        def crooks_once(wF, wR, bandwidth=0.05, grid=(-6,6), ngrid=2000):
            from sklearn.neighbors import KernelDensity
            wR = -wR
            N = min(len(wF), len(wR))
            wF, wR = wF[:N], wR[:N]
            grid = np.linspace(grid[0], grid[1], ngrid)[:,None]
            kdef = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(wF.reshape(-1,1))
            kder = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(wR.reshape(-1,1))
            densf, densr = np.exp(kdef.score_samples(grid)), np.exp(kder.score_samples(grid))
            idx = np.argwhere(np.diff(np.sign(densf - densr)) != 0).reshape(-1)
            if len(idx) == 0:
                return np.nan
            best = idx[np.argmax((densf[idx]+densr[idx])/2)]
            return float(grid[best].ravel()[0])


        estimates = []
        for _ in range(nboot):
            wF_resamp = np.random.choice(wF, len(wF), replace=True)
            wR_resamp = np.random.choice(wR, len(wR), replace=True)
            df = crooks_once(wF_resamp, wR_resamp)
            if not np.isnan(df):
                estimates.append(df)

        if len(estimates) == 0:
            sys.exit("[ERROR] Crooks bootstrap failed (no intersections found)")

        mean_df = float(np.mean(estimates))
        lo, hi = [float(x) for x in np.percentile(estimates, [2.5, 97.5])]

        print(f"[Crooks] ΔF = {mean_df:.8f}")
        print(f"[Crooks] 95% CI = [{lo:.8f}, {hi:.8f}] width={hi-lo:.8f}")

if __name__=="__main__":
    main()
