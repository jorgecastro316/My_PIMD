#!/usr/bin/env python3
# pimc_ho_diagnostics.py
# Combined diagnostics for a 3D isotropic harmonic oscillator sampled via PIMC.
# - Validates energy estimator vs theory and vs bead-XYZ reconstruction.
# - Validates centroid (classical) and bead (quantum) coordinate/radial distributions.
#
# Inputs:
#   --estimator : text file; uses 4th column as potential-energy estimator
#   --centroid  : centroid XYZ-like file (lines "H x y z"; other lines ignored)
#   --bead      : ONE bead XYZ-like file (optional if --bead-files given)
#   --bead-files: MULTIPLE bead XYZ-like files (e.g. bead*.xyz)
#
# Notes:
#   * No seaborn. Only numpy + matplotlib.
#   * Handles --skip (burn-in frames) and --stride (thinning) uniformly across inputs.

import argparse, sys, math, glob
import numpy as np
import matplotlib.pyplot as plt

KB_HA_PER_K = 3.166811563e-6  # Hartree per Kelvin (atomic units)

# ------------------- helpers -------------------

def coth(x):
    # stable coth using numpy
    x = np.asarray(x)
    small = np.abs(x) < 1e-6
    out = np.empty_like(x, dtype=float)
    # series for small x: coth x ≈ 1/x + x/3 - x^3/45
    out[small] = 1.0/x[small] + x[small]/3.0 - (x[small]**3)/45.0
    xs = x[~small]
    out[~small] = np.cosh(xs)/np.sinh(xs)
    return out

def omega_from_K_m(K, m):
    return math.sqrt(K/m)

def sigma2_classical(kT, K):
    # per-coordinate classical variance for HO
    return kT / K

def sigma2_quantum(beta, K, m):
    # per-coordinate quantum variance for HO: σ_q^2 = 1/(2 m ω) coth(β ω / 2)
    w = omega_from_K_m(K, m)
    return (1.0/(2.0*m*w)) * float(coth(0.5*beta*w))

def gaussian_pdf(x, mean, var):
    return (1.0/np.sqrt(2*np.pi*var)) * np.exp(-(x-mean)**2/(2*var))

def maxwell_pdf(r, a):
    # Maxwell with parameter a = per-coordinate stddev
    return np.sqrt(2/np.pi) * (r**2 / a**3) * np.exp(-r**2/(2*a*a))

def gamma_pdf(u, k, theta):
    from math import gamma
    return (u**(k-1)) * np.exp(-u/theta) / (gamma(k) * theta**k)

def is_float(s):
    try:
        float(s); return True
    except Exception:
        return False

def load_xyz_single_atom(fname, skip=0, stride=1):
    """Load single-atom XYZ-like file, collecting lines 'Label x y z'. Apply skip/stride by frames."""
    coords = []
    # We'll treat each 'Label x y z' as a frame; intermediate lines are ignored.
    with open(fname, 'r') as f:
        frames = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and not is_float(parts[0]) and all(is_float(p) for p in parts[1:4]):
                frames.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not frames:
            raise ValueError(f"No coordinate lines found in {fname}. Expect lines like 'H  x  y  z'.")
        frames = np.asarray(frames, dtype=float)
        frames = frames[skip::max(1, stride)]
        return frames

def load_many_bead_files(paths, skip=0, stride=1):
    """Load multiple bead files (one per bead). Returns pooled beads array of shape (n_frames*n_beads, 3)."""
    bead_arrays = []
    for p in paths:
        bead_arrays.append(load_xyz_single_atom(p, skip=0, stride=1))  # load raw; we align after
    # align by shortest length
    n_frames = min(arr.shape[0] for arr in bead_arrays)
    if n_frames == 0:
        raise ValueError("Bead files contain zero frames.")
    beads = np.stack([arr[:n_frames] for arr in bead_arrays], axis=1)  # (n_frames, n_beads, 3)
    # now apply skip/stride by *time frame*
    beads = beads[skip::max(1, stride)]
    if beads.size == 0:
        raise ValueError("No bead frames left after skip/stride.")
    pooled = beads.reshape(-1, 3)
    return pooled

def load_estimator_col4(fname, skip=0, stride=1):
    data = np.loadtxt(fname)
    data = data[skip::max(1, stride)]
    if data.ndim == 1:
        if len(data) < 4:
            raise ValueError(f"Estimator file {fname} has fewer than 4 columns.")
        col4 = data[3]
    else:
        if data.shape[1] < 4:
            raise ValueError(f"Estimator file {fname} has fewer than 4 columns.")
        col4 = data[:,3]
    if col4.size == 0:
        raise ValueError("No estimator samples left after skip/stride.")
    return col4

# ------------------- main -------------------

def main():
    ap = argparse.ArgumentParser(description="PIMC HO combined diagnostics (energy + centroid & bead distributions).")
    ap.add_argument("--estimator", required=True, help="Estimator file (4th column = potential energy estimator)")
    ap.add_argument("--centroid",  required=True, help="Centroid XYZ-like file (single atom)")
    # bead inputs (one of these is optional; you can give both; if both, they are combined)
    ap.add_argument("--bead", help="One bead XYZ-like file (single atom)")
    ap.add_argument("--bead-files", nargs="*", help="Multiple bead files (e.g. bead*.xyz)")
    # physics
    ap.add_argument("--mass", type=float, required=True, help="Mass in atomic units")
    ap.add_argument("--Kspring", type=float, default=1.0, help="Harmonic spring constant K (a.u.)")
    ap.add_argument("--temp", type=float, required=True, help="Temperature in Kelvin")
    # sampling controls
    ap.add_argument("--skip", type=int, default=0, help="Burn-in frames/samples to skip")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame/sample")
    # output
    ap.add_argument("--out", default="pimc_diag.png", help="Output figure filename")
    args = ap.parse_args()

    K = args.Kspring
    m = args.mass
    kT = KB_HA_PER_K * args.temp
    beta = 1.0 / kT
    w = omega_from_K_m(K, m)

    # --- load data
    try:
        V_est = load_estimator_col4(args.estimator, skip=args.skip, stride=args.stride)
    except Exception as e:
        print(f"[Error] reading estimator: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        C = load_xyz_single_atom(args.centroid, skip=args.skip, stride=args.stride)  # (Nc, 3)
    except Exception as e:
        print(f"[Error] reading centroid: {e}", file=sys.stderr)
        sys.exit(1)

    bead_pool = None
    bead_sources = []

    if args.bead:
        try:
            B1 = load_xyz_single_atom(args.bead, skip=args.skip, stride=args.stride)
            bead_sources.append(B1)
        except Exception as e:
            print(f"[Error] reading bead file '{args.bead}': {e}", file=sys.stderr)
            sys.exit(1)

    if args.bead_files:
        paths = []
        for pat in args.bead_files:
            paths.extend(sorted(glob.glob(pat)))
        if paths:
            try:
                B_many = load_many_bead_files(paths, skip=args.skip, stride=args.stride)
                bead_sources.append(B_many)  # already pooled
            except Exception as e:
                print(f"[Error] reading bead-files '{args.bead_files}': {e}", file=sys.stderr)
                sys.exit(1)

    if bead_sources:
        # Pool everything provided: if a single-bead array (N,3), treat it as (N,3) and concatenate;
        # if already pooled (M,3), also concatenate.
        bead_pool = np.vstack(bead_sources) if len(bead_sources) > 1 else bead_sources[0]

    # --- theory
    sigma2_c = sigma2_classical(kT, K)
    sigma_c  = math.sqrt(sigma2_c)
    sigma2_q = sigma2_quantum(beta, K, m)
    sigma_q  = math.sqrt(sigma2_q)
    V_theory_3D = (3*0.25) * w * float(coth(0.5*beta*w))  # 3D quantum expected potential energy

    # --- centroid diagnostics
    x_c, y_c, z_c = C[:,0], C[:,1], C[:,2]
    r_c = np.linalg.norm(C, axis=1)
    var_c_emp = C.var(axis=0, ddof=1).mean()
    r2_c_emp  = np.mean(r_c**2)
    U_c_emp   = 0.5 * K * r2_c_emp  # classical expectation target = 1.5 * kT

    # --- bead diagnostics (if available)
    bead_mode = bead_pool is not None
    if bead_mode:
        x_b, y_b, z_b = bead_pool[:,0], bead_pool[:,1], bead_pool[:,2]
        r_b = np.linalg.norm(bead_pool, axis=1)
        var_b_emp = bead_pool.var(axis=0, ddof=1).mean()
        r2_b_emp  = np.mean(r_b**2)
        U_from_xyz = 0.5 * K * r2_b_emp
    else:
        x_b = y_b = z_b = r_b = None
        var_b_emp = r2_b_emp = U_from_xyz = None

    # --- estimator energy average
    V_est_mean = float(np.mean(V_est))

    # --- print diagnostics
    def relerr(emp, th):
        return (emp - th) / th

    print("=== PIMC HO Combined Diagnostics ===")
    print(f"T = {args.temp} K,  kT = {kT:.6e} Ha,  beta = {beta:.3e} Ha^-1")
    print(f"mass = {m:.6f} a.u.,  K = {K:.6f} a.u.,  omega = {w:.6e} a.u.")
    print(f"samples (estimator) = {len(V_est)},  centroid frames = {len(C)}")
    if bead_mode:
        print(f"pooled bead points  = {len(bead_pool)}")
    print("\n-- Centroid (classical) --")
    print(f"var(coord) emp = {var_c_emp:.6e},   theory (kT/K) = {sigma2_c:.6e},   rel.err = {relerr(var_c_emp, sigma2_c):+.2%}")
    print(f"<r_c^2>   emp = {r2_c_emp:.6e},     theory (3σ^2) = {3.0*sigma2_c:.6e}, rel.err = {relerr(r2_c_emp, 3.0*sigma2_c):+.2%}")
    print(f"<V(centroid)> (from r_c^2) = {U_c_emp:.6e} Ha, theory (3/2 kT) = {1.5*kT:.6e} Ha, rel.err = {relerr(U_c_emp, 1.5*kT):+.2%}")

    print("\n-- Energy (quantum, 3D) --")
    if bead_mode:
        print(f"<V> from bead-XYZ   = {U_from_xyz:.6e} Ha   (should → {V_theory_3D:.6e} Ha)")
    else:
        print("(no bead files provided → skipping <V> from bead-XYZ)")
    print(f"<V> from estimator  = {V_est_mean:.6e} Ha   (should → {V_theory_3D:.6e} Ha)")
    if bead_mode:
        re_xyz = relerr(U_from_xyz, V_theory_3D)
        print(f"rel.err (bead-XYZ vs theory) = {re_xyz:+.2%}")
    re_est = relerr(V_est_mean, V_theory_3D)
    print(f"rel.err (estimator vs theory) = {re_est:+.2%}")

    if bead_mode:
        print("\n-- Beads (quantum) --")
        print(f"var(coord) emp = {var_b_emp:.6e},   theory (σ_q^2) = {sigma2_q:.6e},   rel.err = {relerr(var_b_emp, sigma2_q):+.2%}")
        print(f"<r_b^2>   emp = {r2_b_emp:.6e},     theory (3σ_q^2) = {3.0*sigma2_q:.6e}, rel.err = {relerr(r2_b_emp, 3.0*sigma2_q):+.2%}")

    # --- plots
    rows = 2 if not bead_mode else 2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = np.atleast_2d(axes)

    # 1) centroid coordinate hist + classical Gaussian
    ax = axes[0,0]
    bins = 80
    ax.hist(x_c, bins=bins, density=True, alpha=0.35, label='x_c')
    ax.hist(y_c, bins=bins, density=True, alpha=0.35, label='y_c')
    ax.hist(z_c, bins=bins, density=True, alpha=0.35, label='z_c')
    xs = np.linspace(min(C.min(), -4*sigma_c), max(C.max(), 4*sigma_c), 700)
    ax.plot(xs, gaussian_pdf(xs, 0.0, sigma2_c), lw=2, label='Gaussian theory (centroid)')
    ax.set_title("Centroid coordinate histograms (classical)")
    ax.set_xlabel("value"); ax.set_ylabel("density"); ax.legend()

    # 2) bead coordinate hist + quantum Gaussian
    ax = axes[0,1]
    if bead_mode:
        ax.hist(x_b, bins=bins, density=True, alpha=0.35, label='x_bead')
        ax.hist(y_b, bins=bins, density=True, alpha=0.35, label='y_bead')
        ax.hist(z_b, bins=bins, density=True, alpha=0.35, label='z_bead')
        xb_min = min(bead_pool.min(), -4*sigma_q)
        xb_max = max(bead_pool.max(),  4*sigma_q)
        xgrid = np.linspace(xb_min, xb_max, 700)
        ax.plot(xgrid, gaussian_pdf(xgrid, 0.0, sigma2_q), lw=2, label='Gaussian theory (bead)')
        ax.set_title("Bead coordinate histograms (quantum)")
        ax.set_xlabel("value"); ax.set_ylabel("density"); ax.legend()
    else:
        ax.axis('off')
        ax.set_title("No bead data provided")

    # 3) centroid r + classical Maxwell
    ax = axes[1,0]
    ax.hist(r_c, bins=80, density=True, alpha=0.5, label='r_c empirical')
    rs_c = np.linspace(0, r_c.max()*1.05, 700)
    ax.plot(rs_c, maxwell_pdf(rs_c, sigma_c), lw=2, label='Maxwell theory (centroid)')
    ax.set_title("Centroid radial distribution (classical)")
    ax.set_xlabel("r"); ax.set_ylabel("density"); ax.legend()

    # 4) bead r + quantum Maxwell
    ax = axes[1,1]
    if bead_mode:
        ax.hist(r_b, bins=80, density=True, alpha=0.5, label='r_bead empirical')
        rs_b = np.linspace(0, r_b.max()*1.05, 700)
        ax.plot(rs_b, maxwell_pdf(rs_b, sigma_q), lw=2, label='Maxwell theory (bead)')
        ax.set_title("Bead radial distribution (quantum)")
        ax.set_xlabel("r"); ax.set_ylabel("density"); ax.legend()
    else:
        ax.axis('off')
        ax.set_title("No bead data provided")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"\nSaved figure to {args.out}")

if __name__ == "__main__":
    main()

