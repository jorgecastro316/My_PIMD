#!/usr/bin/env python3
"""
Diagnostics for GPR normal-mode features.

Given a training NPZ (as produced by controller.py --merge-seed / --gpr-train)
and a candidate pool NPZ, this script:
  • Confirms sample counts for the selected direction.
  • Applies the requested feature mode (centroid/raw/nmodes:*).
  • Reports per-mode standard deviations in the normal-mode basis.
  • Fits StandardScaler objects on training features and measures the
    distance from candidates to the training manifold (nearest-neighbour
    distances in scaled feature space).

Use this to understand why predictive uncertainties stay high even with
large training sets (e.g., candidates may still lie far from training
data after the feature transform, or certain modes dominate the metric).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from pimd.utils import rr2nm


# ---------- Helpers ----------

def _reduce_beads(q: np.ndarray, p: np.ndarray, centroid: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce multi-bead arrays to centroid (default) or flatten all beads."""
    if q.ndim == 3:
        if centroid:
            q_red = q.mean(axis=1)
            p_red = p.mean(axis=1)
        else:
            q_red = q.reshape(q.shape[0], -1)
            p_red = p.reshape(p.shape[0], -1)
    elif q.ndim == 2:
        q_red, p_red = q, p
    else:
        raise ValueError(f"Unexpected q shape {q.shape}")
    return q_red, p_red


def select_normal_modes(
    q: np.ndarray, p: np.ndarray, modes: List[int], keep_3d: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform bead coordinates/momenta to normal modes and select subset.

    Parameters
    ----------
    q, p : np.ndarray
        shape (n_samples, nbeads, D)
    modes : list[int]
        indices of normal modes to keep
    keep_3d : bool
        If True, return arrays with mode axis preserved (n_samples, len(modes), D).
    """
    if q.ndim != 3 or p.ndim != 3:
        raise ValueError(f"Expected bead arrays with ndim=3, got q={q.shape}, p={p.shape}")
    n_samples = q.shape[0]
    q_nm = np.array([rr2nm(q[i].T).T for i in range(n_samples)])
    p_nm = np.array([rr2nm(p[i].T).T for i in range(n_samples)])

    q_sel = q_nm[:, modes, :]
    p_sel = p_nm[:, modes, :]
    if keep_3d:
        return q_sel, p_sel
    return q_sel.reshape(n_samples, -1), p_sel.reshape(n_samples, -1)


def apply_feature_mode(
    q: np.ndarray, p: np.ndarray, feature_mode: str, keep_3d: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the same feature-mode logic used in controller.WorkGPRTorch."""
    fm = feature_mode.strip().lower()
    if fm.startswith("nmodes:"):
        _, mode_str = fm.split(":")
        if mode_str.strip() == "all":
            modes = list(range(q.shape[1]))
        else:
            modes = [int(m) for m in mode_str.split(",")]
        return select_normal_modes(q, p, modes, keep_3d=keep_3d)
    if fm == "centroid":
        return _reduce_beads(q, p, centroid=True)
    if fm == "raw":
        return _reduce_beads(q, p, centroid=False)
    raise ValueError(f"Unknown feature_mode '{feature_mode}'")


def sample_rows(arr: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if max_rows <= 0 or len(arr) <= max_rows:
        return arr
    idx = rng.choice(len(arr), size=max_rows, replace=False)
    return arr[idx]


def describe(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


@dataclass
class DiagnosticsResult:
    train_count: int
    cand_count: int
    feature_dim: int
    feature_mode: str
    train_feature_var_stats: Dict[str, float]
    nn_distance_stats: Dict[str, float]
    per_mode_std_q: List[float] | None
    per_mode_std_p: List[float] | None

    def to_dict(self) -> Dict:
        return {
            "train_count": self.train_count,
            "cand_count": self.cand_count,
            "feature_dim": self.feature_dim,
            "feature_mode": self.feature_mode,
            "train_feature_var": self.train_feature_var_stats,
            "nearest_neighbor_distances": self.nn_distance_stats,
            "per_mode_std_q": self.per_mode_std_q,
            "per_mode_std_p": self.per_mode_std_p,
        }


# ---------- Core diagnostics ----------

def run_diagnostics(args: argparse.Namespace) -> DiagnosticsResult:
    rng = np.random.default_rng(args.seed)

    train = np.load(args.train_npz)
    suffix = "F" if args.direction == "forward" else "R"
    q_train = np.array(train[f"q{suffix}"])
    p_train = np.array(train[f"p{suffix}"])
    train = None  # free memory

    cand = np.load(args.cands_npz)
    q_cand = np.array(cand["q"])
    p_cand = np.array(cand["p"])
    cand = None

    q_train_feat, p_train_feat = apply_feature_mode(q_train, p_train, args.feature_mode)
    q_cand_feat, p_cand_feat = apply_feature_mode(q_cand, p_cand, args.feature_mode)

    # Keep 3D data for per-mode stats if using normal modes
    per_mode_std_q = per_mode_std_p = None
    if args.feature_mode.lower().startswith("nmodes:"):
        q_train_nm, p_train_nm = apply_feature_mode(q_train, p_train, args.feature_mode, keep_3d=True)
        per_mode_std_q = q_train_nm.std(axis=(0, 2)).tolist()
        per_mode_std_p = p_train_nm.std(axis=(0, 2)).tolist()

    scaler_q = StandardScaler()
    scaler_p = StandardScaler()
    q_train_scaled = scaler_q.fit_transform(q_train_feat)
    p_train_scaled = scaler_p.fit_transform(p_train_feat)
    train_feat = np.concatenate([q_train_scaled, p_train_scaled], axis=1)

    q_cand_scaled = scaler_q.transform(q_cand_feat)
    p_cand_scaled = scaler_p.transform(p_cand_feat)
    cand_feat = np.concatenate([q_cand_scaled, p_cand_scaled], axis=1)

    train_sample = sample_rows(train_feat, args.max_train, rng)
    cand_sample = sample_rows(cand_feat, args.max_cand, rng)
    dists = pairwise_distances(cand_sample, train_sample, metric="euclidean")
    nn = dists.min(axis=1)

    feat_var = train_feat.var(axis=0)
    feat_var_stats = describe(feat_var)

    return DiagnosticsResult(
        train_count=q_train_feat.shape[0],
        cand_count=q_cand_feat.shape[0],
        feature_dim=train_feat.shape[1],
        feature_mode=args.feature_mode,
        train_feature_var_stats=feat_var_stats,
        nn_distance_stats=describe(nn),
        per_mode_std_q=per_mode_std_q,
        per_mode_std_p=per_mode_std_p,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GPR feature-space diagnostics.")
    parser.add_argument("--train-npz", required=True, help="Path to training NPZ (from gpr_seed merge).")
    parser.add_argument("--cands-npz", required=True, help="Path to candidate pool NPZ.")
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--feature-mode", default="centroid",
                        help="Feature mode (centroid|raw|nmodes:...). Must match training run.")
    parser.add_argument("--max-train", type=int, default=2000,
                        help="Max training samples to use for distance diagnostics (random subset).")
    parser.add_argument("--max-cand", type=int, default=2000,
                        help="Max candidate samples to use for distance diagnostics.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for subsampling.")
    parser.add_argument("--json-out", help="Optional path to dump diagnostics as JSON.")

    args = parser.parse_args()
    result = run_diagnostics(args)

    print("\n=== GPR Feature Diagnostics ===")
    print(f"Training samples ({args.direction}): {result.train_count}")
    print(f"Candidate samples: {result.cand_count}")
    print(f"Feature mode: {result.feature_mode}")
    print(f"Feature dimension after scaling: {result.feature_dim}")
    print("\n[Train feature variance stats]")
    for k, v in result.train_feature_var_stats.items():
        print(f"  {k:>6}: {v:.4f}")
    print("\n[Candidate → Train nearest-neighbour distance stats (scaled feature space)]")
    for k, v in result.nn_distance_stats.items():
        print(f"  {k:>6}: {v:.4f}")

    if result.per_mode_std_q is not None:
        print("\n[Per-mode std dev | positions]")
        for idx, val in enumerate(result.per_mode_std_q):
            print(f"  mode {idx:3d}: {val:.6f}")
    if result.per_mode_std_p is not None:
        print("\n[Per-mode std dev | momenta]")
        for idx, val in enumerate(result.per_mode_std_p):
            print(f"  mode {idx:3d}: {val:.6f}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n[WRITE] Saved diagnostics JSON → {args.json_out}")


if __name__ == "__main__":
    main()
