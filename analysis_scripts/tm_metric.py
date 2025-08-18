#!/usr/bin/env python3
"""
tm_metric.py

Compute Thirumalai–Mountain ergodicity metrics in post-processing from:
  (a) centroid positions (XYZ-style trajectory or flat table),
  (b) an optional energies time series (aligned with centroids).

Formats
-------
Centroids can be provided in two formats:

1) XYZ-style trajectory (like your example)
   Repeated frames:
       <N>
       # time = <float>      (comment line; time optional)
       <sym1> x y z
       <sym2> x y z
       ...
   The script infers N from each frame (must be constant) and ndim=3.

2) Flat table
   Each row = one sample, columns flattened as [x1 y1 z1 x2 y2 z2 ...].
   Provide --natoms and --ndim to interpret columns.

Metrics
-------
Per-particle TM:
  Omega_A(t) = (1/N) * sum_i [ \bar{A}_i(t) - (1/N) * sum_j \bar{A}_j(t) ]^2,
  where \bar{A}_i(t) is the cumulative time average of observable A for particle i.

Single-observable TM:
  chi_A(t) = ( \bar{A}(t) - <A> )^2 for a scalar time series (e.g., potential energy).

Usage examples
--------------
# Your case: XYZ centroids + energies in column 0; burn-in 100; plot and fit slopes
python tm_metric.py --centroids centroids.xyz --centroids-format auto \
    --energies energies.out --energy-col 0 --burn-in 100 \
    --component mag --plot --fit

# Flat table centroids (T x (natoms*ndim)), no energies, 2 header lines, per-atom x
python tm_metric.py --centroids centroids_flat.out --centroids-format flat \
    --natoms 64 --ndim 3 --skip-centroids 2 --component x --plot --fit
"""

import argparse
import numpy as np
import sys
from typing import Tuple, List, Optional

# ----------------------- TM metrics -----------------------

def tm_metric_per_particle(A_ti: np.ndarray, burn_in: int = 0) -> np.ndarray:
    """
    Thirumalai–Mountain metric for a per-particle observable.
    A_ti: array shape (T, N) with T time steps (samples) and N particles.
    Returns: omega_t (length T - burn_in).
    """
    A = np.asarray(A_ti)
    if A.ndim != 2:
        raise ValueError("A_ti must be 2D (T, N).")
    T, N = A.shape
    if not (0 <= burn_in < T):
        raise ValueError(f"Invalid burn_in={burn_in} for T={T}.")
    A = A[burn_in:]
    T = A.shape[0]

    # cumulative time averages per particle: \bar{A}_i(t)
    csum = np.cumsum(A, axis=0, dtype=float)      # (T, N)
    t_idx = np.arange(1, T+1, dtype=float)[:,None]
    barA_ti = csum / t_idx                         # (T, N)

    # mean across particles at each t
    mean_barA_t = barA_ti.mean(axis=1, keepdims=True)  # (T, 1)

    # TM metric Omega_A(t)
    omega_t = np.mean((barA_ti - mean_barA_t)**2, axis=1)  # (T,)
    return omega_t  # expect ~ const/t if ergodic

def tm_metric_single(A_t: np.ndarray, burn_in: int = 0) -> np.ndarray:
    """
    Single-observable variant: chi_A(t) = (barA(t) - meanA)^2.
    A_t: array shape (T,). Returns chi_t (length T - burn_in).
    """
    A = np.asarray(A_t).ravel()
    T = A.shape[0]
    if not (0 <= burn_in < T):
        raise ValueError(f"Invalid burn_in={burn_in} for T={T}.")
    A = A[burn_in:]
    T = A.shape[0]

    csum = np.cumsum(A, dtype=float)
    t_idx = np.arange(1, T+1, dtype=float)
    barA_t = csum / t_idx
    meanA = A.mean()
    chi_t = (barA_t - meanA)**2
    return chi_t  # expect ~ const/t if ergodic

def fit_power_law_slope(y: np.ndarray, eps: float = 1e-16):
    """
    Fit slope of y ~ const * t^s on t=1..len(y) using log10.
    Returns (slope, intercept, r2).
    """
    T = y.shape[0]
    x = np.log10(np.arange(1, T+1, dtype=float))
    yl = np.log10(y + eps)
    X = np.vstack([x, np.ones_like(x)]).T
    theta, *_ = np.linalg.lstsq(X, yl, rcond=None)
    slope, intercept = theta[0], theta[1]
    yhat = X @ theta
    ss_res = np.sum((yl - yhat)**2)
    ss_tot = np.sum((yl - yl.mean())**2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return slope, intercept, r2

# ----------------------- I/O helpers -----------------------

def load_table(path: str, skip: int = 0) -> np.ndarray:
    try:
        return np.loadtxt(path, comments="#", skiprows=skip)
    except Exception as e:
        print(f"Error loading '{path}': {e}", file=sys.stderr)
        sys.exit(1)

def detect_centroids_format(path: str, skip: int) -> str:
    """
    Peek the file to decide 'xyz' vs 'flat':
      - If the first non-empty, non-comment token is an integer (N), assume XYZ.
      - Else assume FLAT.
    """
    with open(path, "r") as f:
        skipped = 0
        for line in f:
            if skipped < skip:
                skipped += 1
                continue
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()[0]
            try:
                int(tok)
                return "xyz"
            except ValueError:
                return "flat"
    # empty file fallback
    return "flat"

def load_centroids_xyz(path: str) -> Tuple[np.ndarray, int, List[float]]:
    """
    Parse an XYZ-style trajectory into array (T, natoms*3).
    Returns (C, natoms, times)
      C: float array shape (T, natoms*3)
      natoms: inferred; must be constant
      times: list of float times if parsable, else NaN
    """
    frames: List[np.ndarray] = []
    times: List[float] = []

    def readline_or_none(f):
        line = f.readline()
        return line if line != "" else None

    with open(path, "r") as f:
        while True:
            # Read number of atoms (skip blank/comment lines until integer found or EOF)
            line = readline_or_none(f)
            while line is not None and (line.strip() == "" or line.lstrip().startswith("#")):
                line = readline_or_none(f)
            if line is None:
                break
            try:
                n = int(line.strip().split()[0])
            except Exception:
                # Not a valid frame start -> stop
                break

            # Read comment/time line
            comment = readline_or_none(f)
            tval = np.nan
            if comment is not None and "#" in comment and "=" in comment:
                try:
                    # parse last token after '='
                    tval = float(comment.split("=")[-1].strip().split()[0])
                except Exception:
                    tval = np.nan
            times.append(tval)

            # Read n atom lines
            coords = []
            for _ in range(n):
                atom_line = readline_or_none(f)
                if atom_line is None:
                    raise ValueError("Unexpected EOF while reading XYZ frame.")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ atom line: '{atom_line.strip()}'")
                # last three tokens assumed to be x y z (ignore symbol and extra props)
                try:
                    x, y, z = map(float, parts[-3:])
                except Exception:
                    raise ValueError(f"Cannot parse coordinates in line: '{atom_line.strip()}'")
                coords.append([x, y, z])

            frames.append(np.asarray(coords, dtype=float))

    if not frames:
        raise ValueError(f"No frames parsed from '{path}'.")

    # Validate constant natoms and flatten
    natoms = frames[0].shape[0]
    for k, fr in enumerate(frames):
        if fr.shape != (natoms, 3):
            raise ValueError(f"Frame {k} shape {fr.shape} differs from first frame {(natoms,3)}.")
    C = np.stack(frames, axis=0).reshape(len(frames), natoms * 3)
    return C, natoms, times

def build_per_atom_observable_from_centroids(C: np.ndarray, natoms: int, ndim: int, component: str) -> np.ndarray:
    """
    C: (T, natoms*ndim) flattened centroid coordinates per sample.
    Returns A_ti with shape (T, natoms) for the chosen component:
      'x', 'y', 'z': pick coordinate
      'mag': Euclidean magnitude per atom
    """
    T, cols = C.shape
    expected = natoms * ndim
    if cols != expected:
        raise ValueError(f"Centroids columns = {cols}, but natoms*ndim = {expected}. "
                         f"Check --natoms/--ndim or input format.")
    C3 = C.reshape(T, natoms, ndim)

    comp = component.lower()
    if comp in ("x", "y", "z"):
        idx = {"x": 0, "y": 1, "z": 2}[comp]
        if idx >= ndim:
            raise ValueError(f"Requested component '{component}' but ndim={ndim}.")
        A_ti = C3[:, :, idx]
    elif comp == "mag":
        A_ti = np.linalg.norm(C3, axis=2)
    else:
        raise ValueError(f"Unknown component '{component}'. Use x|y|z|mag.")
    return A_ti  # (T, natoms)

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Thirumalai–Mountain ergodicity metrics from centroids / energies.")
    ap.add_argument("--centroids", required=True, help="Path to centroids file (XYZ or flat table).")
    ap.add_argument("--centroids-format", choices=["auto", "xyz", "flat"], default="auto",
                    help="Centroids file format: auto-detect, xyz, or flat (default: auto).")
    ap.add_argument("--natoms", type=int, help="Number of atoms (required for flat format).")
    ap.add_argument("--ndim", type=int, default=3, help="Spatial dimensions (default: 3).")
    ap.add_argument("--skip-centroids", type=int, default=0,
                    help="Header lines to skip in centroids file (flat format only).")
    ap.add_argument("--component", choices=["x", "y", "z", "mag"], default="x",
                    help="Per-atom observable from centroids (default: x).")
    ap.add_argument("--burn-in", type=int, default=0, help="Samples to discard from start (default: 0).")

    ap.add_argument("--energies", help="Path to energies file (aligned with centroids).")
    ap.add_argument("--energy-col", type=int, default=0, help="0-based column index in energies file (default: 0).")
    ap.add_argument("--skip-energies", type=int, default=0, help="Header lines to skip in energies file.")

    ap.add_argument("--output-prefix", help="If set, write Omega_A(t)/chi_A(t) to text files with this prefix.")
    ap.add_argument("--plot", action="store_true", help="If set, show log-log plots of metrics.")
    ap.add_argument("--fit", action="store_true", help="If set, fit slope of log10(metric) vs log10(t).")
    args = ap.parse_args()

    # Decide / load centroids
    fmt = args.centroids_format
    if fmt == "auto":
        fmt = detect_centroids_format(args.centroids, args.skip_centroids)
    if fmt == "xyz":
        C, natoms_xyz, times = load_centroids_xyz(args.centroids)
        natoms = natoms_xyz
        ndim = 3
    else:  # flat
        if args.natoms is None:
            raise ValueError("--natoms is required for flat centroids format.")
        natoms = int(args.natoms)
        ndim = int(args.ndim)
        C = load_table(args.centroids, skip=args.skip_centroids)
        # validate column count
        if C.ndim == 1:
            C = C.reshape(1, -1)
        cols = C.shape[1]
        if cols != natoms * ndim:
            raise ValueError(f"Flat centroids columns = {cols}, but natoms*ndim = {natoms*ndim}.")

    # Build per-atom observable from centroids
    A_ti = build_per_atom_observable_from_centroids(C, natoms, ndim, args.component)

    # Per-particle TM metric
    omega_t = tm_metric_per_particle(A_ti, burn_in=args.burn_in)
    print(f"[TM per-particle] length={omega_t.shape[0]} (burn-in={args.burn_in})")
    if args.fit:
        s, b, r2 = fit_power_law_slope(omega_t)
        print(f"  slope ~ {s:.3f} (ideal ergodic: -1.0), R^2={r2:.4f}")

    # Optional single-observable from energies
    chi_t = None
    if args.energies:
        E = load_table(args.energies, skip=args.skip_energies)
        if E.ndim == 1:
            if args.energy_col != 0:
                print(f"Warning: energies is 1D; ignoring --energy-col={args.energy_col}", file=sys.stderr)
            e_series = E
        else:
            if not (0 <= args.energy_col < E.shape[1]):
                raise ValueError(f"--energy-col={args.energy_col} out of range for energies with {E.shape[1]} cols.")
            e_series = E[:, args.energy_col]

        # Align lengths if necessary (truncate to min)
        T_cent = A_ti.shape[0]
        T_eng = e_series.shape[0]
        if T_eng != T_cent:
            T_min = min(T_eng, T_cent)
            print(f"Warning: length mismatch centroids(T={T_cent}) vs energies(T={T_eng}); truncating to {T_min}.",
                  file=sys.stderr)
            e_series = e_series[:T_min]
            omega_t = omega_t[:T_min]  # keep outputs comparable if you like
            A_ti = A_ti[:T_min, :]

        chi_t = tm_metric_single(e_series, burn_in=args.burn_in)
        print(f"[TM single-observable] length={chi_t.shape[0]} (burn-in={args.burn_in})")
        if args.fit:
            sE, bE, r2E = fit_power_law_slope(chi_t)
            print(f"  slope ~ {sE:.3f} (ideal ergodic: -1.0), R^2={r2E:.4f}")

    # Save to files if requested
    if args.output_prefix:
        np.savetxt(f"{args.output_prefix}_TM_per_particle_{args.component}.txt", omega_t)
        if chi_t is not None:
            np.savetxt(f"{args.output_prefix}_TM_single_energy.txt", chi_t)
        print(f"Wrote: {args.output_prefix}_TM_per_particle_{args.component}.txt"
              + (f" and {args.output_prefix}_TM_single_energy.txt" if chi_t is not None else ""))

    # Optional plotting
    if args.plot:
        import matplotlib.pyplot as plt
        # Per-particle TM
        t1 = np.arange(1, omega_t.shape[0] + 1, dtype=float)
        plt.figure()
        plt.loglog(t1, omega_t, label=f"TM per-particle ({args.component})")
        if args.fit:
            s, b, _ = fit_power_law_slope(omega_t)
            c10 = 10**b
            plt.loglog(t1, c10 * (t1 ** s), linestyle="--", label=f"fit: slope={s:.2f}")
        plt.xlabel("t (accepted samples)")
        plt.ylabel(r"$\Omega_A(t)$")
        plt.title("Thirumalai–Mountain metric (per-particle)")
        plt.legend()
        plt.tight_layout()

        # Single observable
        if chi_t is not None:
            t2 = np.arange(1, chi_t.shape[0] + 1, dtype=float)
            plt.figure()
            plt.loglog(t2, chi_t, label="TM single-observable (energy)")
            if args.fit:
                sE, bE, _ = fit_power_law_slope(chi_t)
                c10E = 10**bE
                plt.loglog(t2, c10E * (t2 ** sE), linestyle="--", label=f"fit: slope={sE:.2f}")
            plt.xlabel("t (accepted samples)")
            plt.ylabel(r"$\chi_A(t)$")
            plt.title("Thirumalai–Mountain metric (single observable)")
            plt.legend()
            plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    main()

