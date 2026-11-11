#!/usr/bin/env python3
"""
Aggregate forward/reverse work files and run BAR + Crooks-like diagnostics.

This replacement script intentionally ignores bead-resolved position/velocity
frames so it can post-process the lightweight OpenMM runs that only dump
``*.work.out``.  The interface matches the original CLI:

    python seed_and_crooks.py forward/ reverse/ BETA [--flags ...]

where ``BETA`` is the inverse temperature in atomic units.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


# ---------------------------------------------------------------------------
# I/O helpers


def _find_work_files(root: str, prefix: str, suffix: str) -> Sequence[str]:
    pattern = os.path.join(root, "**", f"{prefix}.{suffix}.out")
    return sorted(glob.glob(pattern, recursive=True))


def _read_work_column(path: str) -> Sequence[float]:
    vals = []
    with open(path) as handle:
        for ln in handle:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                vals.append(float(parts[-1]))
            except ValueError:
                continue
    if not vals:
        raise ValueError(f"{path}: no numeric work values found")
    return vals


def _map_to_unit_interval(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi == lo:
        return np.zeros_like(values, dtype=np.float64)
    return (2.0 * values - (hi + lo)) / (hi - lo)


def _ks_threshold(alpha: float, n: int) -> float:
    if n <= 0:
        return np.inf
    if not (0.0 < alpha < 1.0):
        alpha = 0.05
    return math.sqrt(-0.5 * math.log(alpha / 2.0)) / math.sqrt(n)


def _collect_work(root: str, prefix: str, suffix: str) -> np.ndarray:
    files = _find_work_files(root, prefix, suffix)
    print(f"[INFO] Scanning {root} (pattern {prefix}.{suffix}.out) -> {len(files)} files")
    all_vals: list[float] = []
    for fp in files:
        try:
            all_vals.extend(_read_work_column(fp))
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[WARN] skipping {fp}: {exc}")
    if not all_vals:
        raise SystemExit(f"[ERROR] No work values were found underneath {root}.")
    arr = np.asarray(all_vals, dtype=np.float64)
    print(
        f"[INFO] Loaded {arr.size} samples from {root} "
        f"(mean={arr.mean():.6f}, std={arr.std(ddof=1):.6f})"
    )
    return arr


# ---------------------------------------------------------------------------
# BAR and Crooks utilities (lifted from the previous script but simplified)


def _sigmoid_safe(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def bar_estimate(wf: np.ndarray, wr: np.ndarray, beta: float) -> float:
    wf = np.asarray(wf, float)
    wr = np.asarray(wr, float)
    nF, nR = len(wf), len(wr)
    if nF == 0 or nR == 0:
        raise ValueError("BAR requires both forward and reverse work samples.")
    c = math.log(nF / nR)

    def f(df):
        s1 = sum(_sigmoid_safe(beta * (w - df) - c) for w in wf)
        s2 = sum(_sigmoid_safe(beta * (w + df) + c) for w in wr)
        return s1 - s2

    margin = np.std(np.concatenate([wf, wr])) + 1.0
    lo = min(wf.min(), -wr.max()) - margin
    hi = max(wf.max(), -wr.min()) + margin
    flo, fhi = f(lo), f(hi)
    expand = 0
    while flo * fhi > 0 and expand < 20:
        span = (hi - lo) * 0.5 or 1.0
        lo -= span
        hi += span
        flo, fhi = f(lo), f(hi)
        expand += 1
    if flo * fhi > 0:
        raise RuntimeError("BAR solver could not bracket ΔF; check overlap.")
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=2000)


def bar_ci(wf: np.ndarray, wr: np.ndarray, beta: float, df: float, confidence=0.95):
    nF, nR = len(wf), len(wr)
    c = math.log(nF / nR)
    p = [_sigmoid_safe(beta * (w - df) - c) for w in wf]
    q = [_sigmoid_safe(beta * (w + df) + c) for w in wr]
    info = beta**2 * (sum(pi * (1 - pi) for pi in p) + sum(qj * (1 - qj) for qj in q))
    if info <= 0.0:
        info = 1e-16
    z = norm.ppf(0.5 + confidence / 2.0)
    half = z / math.sqrt(info)
    return (df - half, df + half, 2 * half), info


def _fit_kde(data: np.ndarray, bandwidth: float) -> KernelDensity:
    return KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(data.reshape(-1, 1))


def _find_density_intersection(
    grid: np.ndarray, Pf: np.ndarray, Pr: np.ndarray, *, min_density: float = 1e-6
) -> float:
    diff = Pf - Pr
    density_mask = (Pf > min_density) & (Pr > min_density)
    if not np.any(density_mask):
        return np.nan
    idx = np.where(density_mask)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    segments = []
    start = idx[0]
    for b in breaks:
        segments.append((start, idx[b]))
        start = idx[b + 1]
    segments.append((start, idx[-1]))

    for lo, hi in segments:
        seg_diff = diff[lo : hi + 1]
        if np.all(seg_diff > 0) or np.all(seg_diff < 0):
            continue
        for i in range(lo, hi):
            d1, d2 = diff[i], diff[i + 1]
            if d1 == d2 or d1 * d2 > 0:
                continue
            w1, w2 = grid[i], grid[i + 1]
            return float(w1 - d1 * (w2 - w1) / (d2 - d1))
    return np.nan


@dataclass
class CrooksResult:
    deltaF_crossing: float
    beta_fit_hist: float
    hist_intercept: float
    deltaF_linear_beta: float
    plots: Tuple[str, str]


@dataclass
class ChebyshevResult:
    deltaF: float
    forward_order: int
    reverse_order: int
    forward_ks: float
    reverse_ks: float
    plot: str


def _chebyshev_basis(x: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.size
    T = np.empty((order + 1, n), dtype=np.float64)
    T[0] = 1.0
    if order >= 1:
        T[1] = x
    for k in range(1, order):
        T[k + 1] = 2.0 * x * T[k] - T[k - 1]

    if order == 0:
        U = np.empty((0, n), dtype=np.float64)
        return T, U

    U = np.empty((order, n), dtype=np.float64)
    U[0] = 1.0
    if order >= 2:
        U[1] = 2.0 * x
        for k in range(1, order - 1):
            U[k + 1] = 2.0 * x * U[k] - U[k - 1]
    return T, U


def _fit_chebyshev_ecdf(
    samples: np.ndarray, *, max_order: int, alpha: float
) -> Tuple[np.ndarray, float, float, int, float]:
    values = np.sort(np.asarray(samples, dtype=np.float64))
    n = values.size
    if n == 0:
        raise ValueError("Empty sample array for Chebyshev fit.")

    lo, hi = values[0], values[-1]
    if hi == lo:
        hi = lo + 1.0e-8
    x = np.clip(_map_to_unit_interval(values, lo, hi), -1.0, 1.0)
    ecdf = np.arange(1, n + 1, dtype=np.float64) / n
    threshold = _ks_threshold(alpha, n)

    T, U = _chebyshev_basis(x, max_order)
    d0 = (2.0 / n) * np.sum(np.arccos(x))
    approx = np.full_like(ecdf, d0 / 2.0)
    sqrt_term = np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))

    coeffs = []
    best_deg = 0
    best_dev = np.inf
    selected_dev = np.inf
    selected_deg = max_order

    for j in range(1, max_order + 1):
        U_vals = U[j - 1] if j - 1 < U.shape[0] else np.zeros_like(x)
        coeff = (2.0 / (n * j)) * np.sum(sqrt_term * U_vals)
        coeffs.append(coeff)

        approx += (2.0 / math.pi) * coeff * T[j]
        dev = float(np.max(np.abs(approx - ecdf)))
        if dev < best_dev:
            best_dev = dev
            best_deg = j
        if dev <= threshold:
            selected_deg = j
            selected_dev = dev
            break
    else:
        selected_deg = best_deg
        selected_dev = best_dev

    coeffs = np.asarray(coeffs[:selected_deg], dtype=np.float64)
    return coeffs, lo, hi, selected_deg, selected_dev


def _eval_chebyshev_pdf_from_coeffs(
    coeffs: np.ndarray, lo: float, hi: float, points: np.ndarray
) -> np.ndarray:
    pdf = np.zeros_like(points, dtype=np.float64)
    m = coeffs.size
    if m == 0 or hi == lo:
        return pdf
    mask = (points >= lo) & (points <= hi)
    if not np.any(mask):
        return pdf
    x = np.clip(_map_to_unit_interval(points[mask], lo, hi), -1.0, 1.0)
    _, U = _chebyshev_basis(x, m)
    prefac = 4.0 / (math.pi * (hi - lo))
    pdf_vals = np.zeros_like(x)
    for j in range(1, m + 1):
        pdf_vals += j * coeffs[j - 1] * U[j - 1]
    pdf_vals = np.clip(pdf_vals, 0.0, None)
    pdf[mask] = prefac * pdf_vals
    return pdf


def chebyshev_analysis(
    wf: np.ndarray,
    wr: np.ndarray,
    *,
    grid_pts: int,
    alpha: float,
    max_order: int,
    plot_prefix: str,
) -> ChebyshevResult:
    coeffF, loF, hiF, degF, ksF = _fit_chebyshev_ecdf(wf, max_order=max_order, alpha=alpha)
    coeffR, loR, hiR, degR, ksR = _fit_chebyshev_ecdf(-wr, max_order=max_order, alpha=alpha)

    grid_lo = min(loF, loR)
    grid_hi = max(hiF, hiR)
    span = max(grid_hi - grid_lo, 1e-6)
    grid = np.linspace(grid_lo - 0.25 * span, grid_hi + 0.25 * span, grid_pts)

    Pf = _eval_chebyshev_pdf_from_coeffs(coeffF, loF, hiF, grid)
    Pr = _eval_chebyshev_pdf_from_coeffs(coeffR, loR, hiR, grid)
    deltaF = _find_density_intersection(grid, Pf, Pr)

    plot_path = f"{plot_prefix}_chebyshev.png"
    plt.figure(figsize=(9, 5))
    plt.plot(grid, Pf, label="Forward ECDF fit", color="tab:red")
    plt.plot(grid, Pr, label="Reverse ECDF fit (-W)", color="tab:blue")
    if np.isfinite(deltaF):
        plt.axvline(deltaF, linestyle="--", color="k", label="Intersection")
    plt.xlabel("Work / Ha")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return ChebyshevResult(
        deltaF=float(deltaF) if np.isfinite(deltaF) else float("nan"),
        forward_order=degF,
        reverse_order=degR,
        forward_ks=ksF,
        reverse_ks=ksR,
        plot=plot_path,
    )


def crooks_analysis(
    wf: np.ndarray,
    wr: np.ndarray,
    *,
    beta: float,
    bandwidth: float,
    grid_pts: int,
    log_bins: int,
    plot_prefix: str,
) -> CrooksResult:
    forward = np.asarray(wf, float)
    reverse = -np.asarray(wr, float)

    combined = np.concatenate([forward, reverse])
    xmin, xmax = combined.min(), combined.max()
    span = max(xmax - xmin, 1e-6)
    grid = np.linspace(xmin - 0.25 * span, xmax + 0.25 * span, grid_pts).reshape(-1, 1)
    grid_flat = grid[:, 0]

    kdef = _fit_kde(forward, bandwidth)
    kder = _fit_kde(reverse, bandwidth)
    logPf = kdef.score_samples(grid)
    logPr = kder.score_samples(grid)
    Pf, Pr = np.exp(logPf), np.exp(logPr)

    deltaF_crossing = _find_density_intersection(grid_flat, Pf, Pr)
    if np.isnan(deltaF_crossing):
        print("[CROOKS] No meaningful KDE intersection (insufficient overlap).")

    counts_f, edges = np.histogram(forward, bins=log_bins, range=(xmin, xmax))
    counts_r, _ = np.histogram(reverse, bins=edges)
    widths = np.diff(edges)
    dens_f = counts_f / (len(forward) * widths)
    dens_r = counts_r / (len(reverse) * widths)
    centers = edges[:-1] + widths / 2.0
    mask = (dens_f > 0) & (dens_r > 0)
    log_ratio = np.full_like(centers, np.nan, dtype=float)
    log_ratio[mask] = np.log(dens_f[mask] / dens_r[mask])

    beta_fit_hist = np.nan
    hist_intercept = np.nan
    deltaF_linear_beta = np.nan
    if np.count_nonzero(mask) > 1:
        slope, intercept = np.polyfit(centers[mask], log_ratio[mask], 1)
        beta_fit_hist = slope
        hist_intercept = intercept

    # Force a Crooks line with the provided beta: log Pf/Pr = beta*(W - ΔF)
    if np.count_nonzero(mask) > 0:
        intercept_fixed = np.nanmean(log_ratio[mask] - beta * centers[mask])
        deltaF_linear_beta = -intercept_fixed / beta

    plot_dir = os.path.dirname(plot_prefix)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(grid[:, 0], Pf, label="Forward KDE", color="tab:red")
    plt.plot(grid[:, 0], Pr, label="Reverse KDE (-W)", color="tab:blue")
    if not np.isnan(deltaF_crossing):
        plt.axvline(deltaF_crossing, color="k", linestyle="--", label="Intersection")
    plt.xlabel("Work / Ha")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    dist_path = f"{plot_prefix}_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(centers[mask], log_ratio[mask], s=18, color="tab:green", label="Histogram")
    if np.isfinite(beta_fit_hist):
        xs = np.linspace(centers[mask].min(), centers[mask].max(), 128)
        plt.plot(xs, beta_fit_hist * xs + hist_intercept, "--", color="k", label=f"Fit β={beta_fit_hist:.3f}")
    if np.isfinite(deltaF_linear_beta):
        xs = np.linspace(centers[mask].min(), centers[mask].max(), 128)
        plt.plot(xs, beta * (xs - deltaF_linear_beta), ":", color="tab:red", label=f"Fixed β line (ΔF={deltaF_linear_beta:.3e})")
    plt.legend()
    plt.xlabel("Work / Ha")
    plt.ylabel("log P_F(W) - log P_R(-W)")
    plt.tight_layout()
    ratio_path = f"{plot_prefix}_logratio.png"
    plt.savefig(ratio_path, dpi=200)
    plt.close()

    return CrooksResult(
        deltaF_crossing=float(deltaF_crossing),
        beta_fit_hist=float(beta_fit_hist),
        hist_intercept=float(hist_intercept),
        deltaF_linear_beta=float(deltaF_linear_beta),
        plots=(dist_path, ratio_path),
    )


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate work files and perform BAR + Crooks analysis."
    )
    parser.add_argument("forward_root")
    parser.add_argument("reverse_root")
    parser.add_argument("beta", type=float, help="Inverse temperature β (Ha^-1)")
    parser.add_argument("--prefix", default="simulation", help="Work file prefix (default: simulation)")
    parser.add_argument(
        "--work-suffix",
        default="work",
        help="Suffix inside the *.out files (default: work)",
    )
    parser.add_argument(
        "--max-per-dir",
        type=int,
        default=None,
        help="Optional number of samples to keep per direction (after shuffling).",
    )
    parser.add_argument(
        "--plot-prefix",
        default="plots/crooks",
        help="Prefix for the PNG plots.",
    )
    parser.add_argument(
        "--out",
        default="data/seed_work.npz",
        help="NPZ file where the aggregated work arrays will be stored.",
    )
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=0.02,
        help="Gaussian KDE bandwidth for Crooks analysis.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=2000,
        help="Number of grid points for KDE evaluation.",
    )
    parser.add_argument(
        "--log-bins",
        type=int,
        default=400,
        help="Histogram bins for the log-ratio diagnostic.",
    )
    parser.add_argument(
        "--cheb-max-order",
        type=int,
        default=40,
        help="Maximum Chebyshev polynomial order for ECDF smoothing.",
    )
    parser.add_argument(
        "--cheb-alpha",
        type=float,
        default=0.05,
        help="Significance level for the KS test used to choose Chebyshev order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wf = _collect_work(args.forward_root, args.prefix, args.work_suffix)
    wr = _collect_work(args.reverse_root, args.prefix, args.work_suffix)

    if args.max_per_dir is not None:
        n_keep = min(args.max_per_dir, wf.size, wr.size)
        print(f"[INFO] Truncating to {n_keep} samples per direction for analysis.")
        wf = wf[:n_keep]
        wr = wr[:n_keep]

    beta = float(args.beta)
    seed = {"wF": wf, "wR": wr, "beta": beta}

    # Save aggregated work arrays for downstream scripts.
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, wF=wf, wR=wr, beta=beta)
    print(f"[SAVE] wrote {args.out}")

    # Linear-response estimates (valid when near-quasistatic): ΔF ≈ 0.5(<Wf> - <Wr>), diss ≈ 0.5(<Wf> + <Wr>)
    mean_wf = wf.mean()
    mean_wr = wr.mean()
    df_lr = 0.5 * (mean_wf - mean_wr)
    diss_lr = 0.5 * (mean_wf + mean_wr)
    # simple standard errors assuming iid
    se_wf = wf.std(ddof=1) / math.sqrt(len(wf))
    se_wr = wr.std(ddof=1) / math.sqrt(len(wr))
    se_df_lr = 0.5 * math.sqrt(se_wf**2 + se_wr**2)
    se_diss_lr = 0.5 * math.sqrt(se_wf**2 + se_wr**2)
    print(f"[LINRESP] ΔF_lr = {df_lr:.8f} ± {se_df_lr:.8f}")
    print(f"[LINRESP] diss_lr = {diss_lr:.8f} ± {se_diss_lr:.8f}")

    # BAR estimate
    df_bar = bar_estimate(wf, wr, beta)
    ci_bar, fisher = bar_ci(wf, wr, beta, df_bar)
    print(f"[BAR] ΔF = {df_bar:.8f}  (95% CI: [{ci_bar[0]:.8f}, {ci_bar[1]:.8f}], width={ci_bar[2]:.8f})")
    print(f"[BAR] Fisher information = {fisher:.3f}")

    crooks = crooks_analysis(
        wf,
        wr,
        beta=beta,
        bandwidth=args.kde_bandwidth,
        grid_pts=args.grid_points,
        log_bins=args.log_bins,
        plot_prefix=args.plot_prefix,
    )
    print(
        "[CROOKS] ΔF intersection = "
        f"{crooks.deltaF_crossing:.8f}, "
        f"β_fit(hist)={crooks.beta_fit_hist:.6f}"
    )
    if np.isfinite(crooks.deltaF_linear_beta):
        print(f"[CROOKS] ΔF from fixed β line (β={beta:.4f}) = {crooks.deltaF_linear_beta:.8f}")
    print(f"[CROOKS] Saved plots -> {crooks.plots}")

    cheb = chebyshev_analysis(
        wf,
        wr,
        grid_pts=args.grid_points,
        alpha=args.cheb_alpha,
        max_order=args.cheb_max_order,
        plot_prefix=args.plot_prefix,
    )
    print(
        "[CHEB] forward_deg={:d} (KS={:.3e}), reverse_deg={:d} (KS={:.3e})".format(
            cheb.forward_order, cheb.forward_ks, cheb.reverse_order, cheb.reverse_ks
        )
    )
    if np.isfinite(cheb.deltaF):
        print(f"[CHEB] ΔF (ECDF intersection) = {cheb.deltaF:.8f}")
    else:
        print("[CHEB] No ECDF intersection detected (insufficient overlap).")
    print(f"[CHEB] Saved plot -> {cheb.plot}")


if __name__ == "__main__":
    main()
