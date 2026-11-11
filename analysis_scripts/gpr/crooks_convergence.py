#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.neighbors import KernelDensity
import multiprocessing as mp, os

def crooks_once(wF, wR, bandwidth=0.05, grid=(-6,6), ngrid=2000):
    """Single Crooks ΔF estimate from forward/reverse work arrays"""
    wR = -wR
    N = min(len(wF), len(wR))
    wF, wR = wF[:N], wR[:N]
    grid = np.linspace(grid[0], grid[1], ngrid)[:, None]
    kdef = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(wF.reshape(-1, 1))
    kder = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(wR.reshape(-1, 1))
    densf, densr = np.exp(kdef.score_samples(grid)), np.exp(kder.score_samples(grid))
    idx = np.argwhere(np.diff(np.sign(densf - densr)) != 0).reshape(-1)
    if len(idx) == 0:
        return np.nan
    best = idx[np.argmax((densf[idx] + densr[idx]) / 2)]
    return float(grid[best].ravel()[0])

# --- Global worker so multiprocessing can pickle it ---
def crooks_once_seed(seed, wF, wR):
    rng = np.random.default_rng(seed)
    wF_resamp = rng.choice(wF, len(wF), replace=True)
    wR_resamp = rng.choice(wR, len(wR), replace=True)
    return crooks_once(wF_resamp, wR_resamp)

def crooks_bootstrap(wF, wR, nboot=200, nprocs=1):
    """Bootstrap Crooks ΔF with parallelization"""
    args = [(seed, wF, wR) for seed in range(nboot)]
    with mp.Pool(processes=nprocs) as pool:
        estimates = pool.starmap(crooks_once_seed, args)

    estimates = [df for df in estimates if not np.isnan(df)]
    if len(estimates) == 0:
        raise RuntimeError("Crooks bootstrap failed (no intersections found)")
    mean_df = float(np.mean(estimates))
    lo, hi = [float(x) for x in np.percentile(estimates, [2.5, 97.5])]
    return mean_df, lo, hi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crooks ΔF convergence analysis")
    parser.add_argument("--seed", type=str, required=True, help="Final converged seed .npz")
    parser.add_argument("--bootstrap", type=int, default=200, help="Number of bootstrap replicates")
    parser.add_argument("--step", type=int, default=1000, help="Step size for subsampling")
    args = parser.parse_args()

    if not os.path.exists(args.seed):
        raise FileNotFoundError(f"Seed file not found: {args.seed}")

    data = np.load(args.seed, allow_pickle=True)
    wF, wR = np.asarray(data["wF"], float), np.asarray(data["wR"], float)

    nprocs = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    print(f"[INFO] Using {nprocs} cores for bootstrap")

    sizes, means, lows, highs = [], [], [], []
    for N in range(args.step, min(len(wF), len(wR)), args.step):
        df, lo, hi = crooks_bootstrap(wF[:N], wR[:N], nboot=args.bootstrap, nprocs=nprocs)
        sizes.append(N); means.append(df); lows.append(lo); highs.append(hi)
        print(f"[Crooks] N={N} ΔF={df:.6f} [{lo:.6f}, {hi:.6f}]")

    np.savez("crooks_convergence.npz",
             sizes=np.array(sizes),
             mean=np.array(means),
             lo=np.array(lows),
             hi=np.array(highs))

    plt.figure(figsize=(10,6))
    plt.fill_between(sizes, lows, highs, alpha=0.3, label="95% CI (bootstrap)")
    plt.plot(sizes, means, color="r", lw=2, label="Crooks ΔF")
    plt.axhline(-2.97, ls="--", color="k", label="Analytical reference")
    plt.xlabel("Number of data points")
    plt.ylabel("ΔF [Ha]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("crooks_convergence.png", dpi=150)
    plt.show()
