#!/usr/bin/env python3
# story_pool.py
import os, argparse, numpy as np, matplotlib.pyplot as plt

def load_pool(path):
    d = np.load(path, allow_pickle=True)
    q = d["q"]  # (n_samples, nbeads, natoms*ndim)
    p = d["p"]  # same shape
    natoms = int(d["natoms"]); ndim = int(d["ndim"]); nbeads = int(d["nbeads"])
    return q, p, natoms, ndim, nbeads

def bead_coord_series(q, coord_idx):
    # q: (S, B, C); return (S, B) for chosen coord
    return q[:, :, coord_idx]

def centroid_over_beads(arr_bead):
    # arr_bead: (S, B) -> (S,)
    return arr_bead.mean(axis=1)

def width_over_beads(arr_bead):
    # std across beads per sample: (S,)
    return arr_bead.std(axis=1)

def rolling_stats(x, win):
    if win <= 1: return x, np.zeros_like(x)
    # simple moving average & std with convolution
    k = np.ones(win) / win
    mean = np.convolve(x, k, mode="same")
    # rolling std (approx): E[x^2]-E[x]^2
    x2 = np.convolve(x*x, k, mode="same")
    var = np.clip(x2 - mean*mean, 0, None)
    return mean, np.sqrt(var)

def acf(x, max_lag=100):
    x = np.asarray(x); x = x - x.mean()
    n = len(x); var = np.dot(x, x)
    if var == 0: return np.zeros(max_lag+1)
    ac = np.empty(max_lag+1)
    for k in range(max_lag+1):
        ac[k] = np.dot(x[:n-k], x[k:]) / var
    return ac

def is_all_zero(x, atol=0.0):
    return np.all(np.isclose(x, 0.0, atol=atol))

def main():
    ap = argparse.ArgumentParser(description="Time-story plots for a PIMC pool (1D view on a chosen coordinate).")
    ap.add_argument("--pool", required=True, help=".npz file with q,p,natoms,ndim,nbeads")
    ap.add_argument("--coord", type=int, default=0, help="Flattened coord index in [0, natoms*ndim)")
    ap.add_argument("--outdir", default="figs_story", help="Output directory for figures")
    ap.add_argument("--roll", type=int, default=101, help="Rolling window (odd) for smoothing (0/1 disables)")
    ap.add_argument("--max-acf", type=int, default=200, help="Max lag for ACF")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    q, p, natoms, ndim, nbeads = load_pool(args.pool)
    S, B, C = q.shape
    if not (0 <= args.coord < C):
        raise SystemExit(f"--coord {args.coord} out of range (C={C})")

    # Extract the chosen flattened coordinate across beads
    Xb = bead_coord_series(q, args.coord)  # (S, B)
    Xc = centroid_over_beads(Xb)           # (S,)
    W  = width_over_beads(Xb)              # (S,)

    # Momentum centroid if present
    Pb = bead_coord_series(p, args.coord)  # (S, B)
    Pc = centroid_over_beads(Pb)           # (S,)
    zeros_p = is_all_zero(Pc)

    # Diagnostics
    drift = Xc[-1] - Xc[0]
    diffs = np.diff(Xc)
    big_jump_thr = 5.0 * np.std(diffs) if diffs.size > 5 else np.inf
    big_jumps = np.where(np.abs(diffs) > max(big_jump_thr, 1e-14))[0]  # indices where jump occurs between i->i+1
    # sticking: low std in windows
    win = max(25, args.roll if args.roll and args.roll > 1 else 101)
    _, xsig = rolling_stats(Xc, win)
    stuck_idx = np.where(xsig < (0.05 * np.std(Xc) if np.std(Xc) > 0 else 0))[0]

    print(f"[INFO] Samples={S}, Beads={B}, Coords={C} (natoms*ndim)")
    print(f"[INFO] Drift (Xc[last]-Xc[first]) = {drift:.6g}")
    print(f"[INFO] Big jumps (>~5σ of ΔX): count={len(big_jumps)} at indices={big_jumps[:10]}{'...' if len(big_jumps)>10 else ''}")
    print(f"[INFO] Stuck (rolling σ small): ~{len(stuck_idx)} samples flagged")
    if zeros_p:
        print(f"[INFO] Momentum centroid is all zeros (likely no velocities were generated).")

    # 1) Time series of centroid position
    plt.figure()
    t = np.arange(S)
    plt.plot(t, Xc, lw=0.6, label="centroid q")
    if args.roll and args.roll > 1 and args.roll % 2 == 1:
        m, s = rolling_stats(Xc, args.roll)
        plt.plot(t, m, lw=1.5, label=f"rolling mean (win={args.roll})")
        plt.fill_between(t, m - 1.96*s/np.sqrt(max(1,args.roll)), m + 1.96*s/np.sqrt(max(1,args.roll)),
                         alpha=0.2, label="~95% CI (roll)")
    if len(big_jumps) > 0:
        plt.scatter(big_jumps, Xc[big_jumps], s=12, color='r', label="big jump")
    plt.xlabel("sample index")
    plt.ylabel("centroid q (chosen coord)")
    plt.title("Centroid coordinate over accepted samples")
    plt.legend()
    f1 = os.path.join(args.outdir, "01_centroid_q_time.png")
    plt.tight_layout(); plt.savefig(f1, dpi=args.dpi); plt.close(); print(f"[SAVE] {f1}")

    # 2) Bead-by-time heatmap (chosen coord)
    plt.figure()
    plt.imshow(Xb.T, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="q (chosen coord)")
    plt.xlabel("sample index"); plt.ylabel("bead index")
    plt.title("Bead positions (chosen coord) over accepted samples")
    f2 = os.path.join(args.outdir, "02_bead_heatmap.png")
    plt.tight_layout(); plt.savefig(f2, dpi=args.dpi); plt.close(); print(f"[SAVE] {f2}")

    # 3) Ring-polymer width vs time
    plt.figure()
    plt.plot(t, W, lw=0.7, label="RP width (std across beads)")
    if args.roll and args.roll > 1 and args.roll % 2 == 1:
        mW, sW = rolling_stats(W, args.roll)
        plt.plot(t, mW, lw=1.2, label="rolling mean")
    plt.xlabel("sample index"); plt.ylabel("std over beads")
    plt.title("Ring-polymer width (chosen coord)")
    plt.legend()
    f3 = os.path.join(args.outdir, "03_width_time.png")
    plt.tight_layout(); plt.savefig(f3, dpi=args.dpi); plt.close(); print(f"[SAVE] {f3}")

    # 4) Momentum centroid vs time (if any)
    if not zeros_p:
        plt.figure()
        plt.plot(t, Pc, lw=0.6, label="centroid p")
        if args.roll and args.roll > 1 and args.roll % 2 == 1:
            mp, sp = rolling_stats(Pc, args.roll)
            plt.plot(t, mp, lw=1.5, label="rolling mean")
        plt.xlabel("sample index"); plt.ylabel("centroid p (chosen coord)")
        plt.title("Centroid momentum over accepted samples")
        plt.legend()
        f4 = os.path.join(args.outdir, "04_centroid_p_time.png")
        plt.tight_layout(); plt.savefig(f4, dpi=args.dpi); plt.close(); print(f"[SAVE] {f4}")

    # 5) Autocorrelation of centroid q
    plt.figure()
    ac = acf(Xc, max_lag=args.max_acf)
    lags = np.arange(len(ac))
    plt.stem(lags, ac, use_line_collection=True, basefmt=" ")
    plt.xlabel("lag"); plt.ylabel("ACF")
    plt.title("Autocorrelation of centroid q")
    f5 = os.path.join(args.outdir, "05_acf_q.png")
    plt.tight_layout(); plt.savefig(f5, dpi=args.dpi); plt.close(); print(f"[SAVE] {f5}")

if __name__ == "__main__":
    main()
