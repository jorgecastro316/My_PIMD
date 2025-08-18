#!/usr/bin/env python3
# visualize_classical_HO.py
import argparse, sys, math
import numpy as np
import matplotlib.pyplot as plt

KB_HA_PER_K = 3.166811563e-6  # Hartree per Kelvin (atomic units)

def is_float(s):
    try:
        float(s)
        return True
    except Exception:
        return False

def load_single_atom_xyz_positions(path):
    coords = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and not is_float(parts[0]) and all(is_float(p) for p in parts[1:4]):
                x, y, z = map(float, parts[1:4])
                coords.append([x, y, z])
    if not coords:
        raise ValueError("No coordinate lines found.")
    return np.asarray(coords, dtype=float)

def gaussian_pdf(x, mean, var):
    return (1.0/np.sqrt(2*np.pi*var)) * np.exp(-(x-mean)**2/(2*var))

def maxwell_pdf(r, a):
    return np.sqrt(2/np.pi) * (r**2 / a**3) * np.exp(-r**2/(2*a*a))

def gamma_pdf(u, k, theta):
    from math import gamma
    return (u**(k-1)) * np.exp(-u/theta) / (gamma(k) * theta**k)

def qqplot_gaussian(ax, data, mean, var):
    data = np.sort(data)
    q = np.linspace(0, 1, len(data), endpoint=False)[1:]
    from scipy.stats import norm
    theo = mean + np.sqrt(var) * norm.ppf(q)
    ax.plot(theo, data, 'o', markersize=2, alpha=0.5)
    lims = [min(theo.min(), data.min()), max(theo.max(), data.max())]
    ax.plot(lims, lims, 'k-', lw=1)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("QQ-plot (Gaussian)")

def main():
    ap = argparse.ArgumentParser(description="Visual diagnostics for classical 3D isotropic HO (single bead).")
    ap.add_argument("xyz", help="Path to XYZ file")
    ap.add_argument("--Kspring", type=float, default=1.0, help="Spring constant K in a.u. (default 1.0)")
    ap.add_argument("--temp", type=float, required=True, help="Temperature in Kelvin")
    ap.add_argument("--skip", type=int, default=0, help="Burn-in frames to skip")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    ap.add_argument("--qq", action="store_true", help="Add QQ-plot for one coordinate")
    ap.add_argument("--out", default="classical_HO_diagnostics.png", help="Output image file")
    args = ap.parse_args()

    Q = load_single_atom_xyz_positions(args.xyz)
    Q = Q[args.skip::max(1, args.stride)]
    if Q.size == 0:
        sys.exit("No samples left after skip/stride.")

    kT = KB_HA_PER_K * args.temp
    K = args.Kspring
    sigma2 = kT / K
    sigma = math.sqrt(sigma2)

    x, y, z = Q[:,0], Q[:,1], Q[:,2]
    r = np.linalg.norm(Q, axis=1)
    U = 0.5 * K * r**2

    nrows = 3 if args.qq else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5.5 + 2.5*(nrows-2)))
    axes = np.atleast_2d(axes)

    # 1) Coordinate histograms
    ax = axes[0,0]
    bins = 80
    ax.hist(x, bins=bins, density=True, alpha=0.35, label='x')
    ax.hist(y, bins=bins, density=True, alpha=0.35, label='y')
    ax.hist(z, bins=bins, density=True, alpha=0.35, label='z')
    xs = np.linspace(min(Q.min(), -3*sigma), max(Q.max(), 3*sigma), 500)
    ax.plot(xs, gaussian_pdf(xs, 0.0, sigma2), 'k-', lw=2, label='Gaussian theory')
    ax.set_title("Coordinate histograms + Gaussian")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

    # 2) x-y scatter + 1σ circle
    ax = axes[0,1]
    take = max(1, len(x)//50000)
    ax.scatter(x[::take], y[::take], s=2, alpha=0.4)
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(sigma*np.cos(theta), sigma*np.sin(theta), 'r-', lw=2)
    ax.set_aspect('equal', 'box')
    ax.set_title("x vs y scatter + 1σ circle")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # 3) r histogram + Maxwell overlay
    ax = axes[1,0]
    ax.hist(r, bins=80, density=True, alpha=0.5, label='Empirical r')
    rs = np.linspace(0, r.max()*1.05, 500)
    ax.plot(rs, maxwell_pdf(rs, sigma), 'k-', lw=2, label='Maxwell theory')
    ax.set_title("r histogram + Maxwell")
    ax.set_xlabel("r")
    ax.set_ylabel("Density")
    ax.legend()

    # 4) U histogram + Gamma overlay
    ax = axes[1,1]
    ax.hist(U, bins=80, density=True, alpha=0.5, label='Empirical U')
    us = np.linspace(0, U.max()*1.05, 500)
    ax.plot(us, gamma_pdf(us, k=1.5, theta=kT), 'k-', lw=2, label='Gamma theory')
    ax.set_title("U histogram + Gamma(k=3/2, θ=kT)")
    ax.set_xlabel("Potential Energy (Ha)")
    ax.set_ylabel("Density")
    ax.legend()

    # Optional QQ-plot
    if args.qq:
        ax = axes[2,0]
        qqplot_gaussian(ax, x, 0.0, sigma2)
        axes[2,1].axis('off')

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved diagnostics to {args.out}")

if __name__ == "__main__":
    main()

