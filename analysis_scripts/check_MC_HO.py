#!/usr/bin/env python3
# analyze_xyz_classical_HO.py
import argparse, sys, math
import numpy as np

KB_HA_PER_K = 3.166811563e-6  # Hartree/K (CODATA a.u.)

def is_float(s):
    try:
        float(s)
        return True
    except Exception:
        return False

def load_single_atom_xyz_positions(path):
    """
    Parse single-atom XYZ-like file where frames are:
        H x y z
        1
        # time = ...
    Only collects lines that look like: <label> <float> <float> <float>.
    Returns: (n_frames, 3) numpy array.
    """
    coords = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and not is_float(parts[0]) and all(is_float(p) for p in parts[1:4]):
                try:
                    x, y, z = map(float, parts[1:4])
                    coords.append([x, y, z])
                except Exception:
                    continue
    if not coords:
        raise ValueError("No coordinate lines found.")
    return np.asarray(coords, dtype=float)

def analyze_classical_HO(Q, mass, Kspring, temperature, kb=KB_HA_PER_K):
    """
    Validate single-bead (classical) MC for isotropic HO using spring constant K.
    Q: (n_samples, 3) array for one atom in 3D.
    """
    if Q.ndim != 2 or Q.shape[1] != 3:
        raise ValueError(f"Q must be (n_samples, 3); got {Q.shape}")

    n_samples = Q.shape[0]
    dof = 3  # single atom in 3D
    kT = kb * temperature
    omega = math.sqrt(Kspring / mass)

    # Exact expectations
    sigma2_expected = kT / Kspring             # variance per coordinate
    r2_expected = dof * sigma2_expected
    V_expected = 0.5 * Kspring * r2_expected
    V_equip = 0.5 * dof * kT

    # Empirical stats
    coord_means = Q.mean(axis=0)
    coord_vars = Q.var(axis=0, ddof=1)
    r2_samples = np.sum(Q**2, axis=1)
    V_samples = 0.5 * Kspring * r2_samples

    mean_abs_coord_mean = np.mean(np.abs(coord_means))
    var_mean = coord_vars.mean()
    r2_mean = r2_samples.mean()
    V_mean = V_samples.mean()

    def se_mean(x):
        return np.asarray(x).std(ddof=1)/math.sqrt(len(x))
    se_var_mean = coord_vars.std(ddof=1)/math.sqrt(len(coord_vars))
    se_r2_mean = se_mean(r2_samples)
    se_V_mean = se_mean(V_samples)

    relerr_var = (var_mean - sigma2_expected)/sigma2_expected
    relerr_r2 = (r2_mean - r2_expected)/r2_expected
    relerr_V = (V_mean - V_expected)/V_expected

    print("=== Classical 3D Isotropic HO — Validation (single bead) ===")
    print(f"samples                  : {n_samples}")
    print(f"T (K), kT (Ha)           : {temperature:.6g}, {kT:.6e}")
    print(f"mass (au)                : {mass:.6g}")
    print(f"K_spring (au)            : {Kspring:.6g}")
    print(f"omega (au)               : {omega:.6g}")

    print("\n-- Coordinate level checks --")
    print(f"mean(|coord means|)      : {mean_abs_coord_mean:.3e}  (should be ~0)")
    print(f"coord vars (emp)         : {coord_vars}")
    print(f"mean coord var (emp)     : {var_mean:.6e} ± {se_var_mean:.1e}")
    print(f"mean coord var (exact)   : {sigma2_expected:.6e}")
    print(f"rel. error var           : {relerr_var:+.3%}")

    print("\n-- r^2 and Potential Energy --")
    print(f"<r^2> emp                : {r2_mean:.6e} ± {se_r2_mean:.1e}")
    print(f"<r^2> exact              : {r2_expected:.6e}")
    print(f"rel. error <r^2>         : {relerr_r2:+.3%}")

    print(f"<V> emp (Ha)             : {V_mean:.6e} ± {se_V_mean:.1e}")
    print(f"<V> exact (Ha)           : {V_expected:.6e}")
    print(f"rel. error <V>           : {relerr_V:+.3%}")

    print("\n-- Virial / Equipartition sanity --")
    print(f"Equipartition V_exact    : {V_equip:.6e}  (should equal exact <V>)")
    print("For classical HO: <V> = (d/2) kT and <K> = (d/2) kT, so total <E> = d kT.")

def main():
    ap = argparse.ArgumentParser(description="Validate single-bead (classical) MC for 3D isotropic HO from a single-atom XYZ.")
    ap.add_argument("xyz", help="Path to the XYZ file of accepted configurations")
    ap.add_argument("--mass", type=float, required=True, help="Mass in atomic units")
    ap.add_argument("--Kspring", type=float, default=1.0, help="Spring constant K in atomic units (default 1.0)")
    ap.add_argument("--temp", type=float, required=True, help="Temperature in Kelvin")
    ap.add_argument("--skip", type=int, default=0, help="Burn-in frames to skip")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    args = ap.parse_args()

    try:
        Q = load_single_atom_xyz_positions(args.xyz)
    except Exception as e:
        print(f"Error reading positions: {e}", file=sys.stderr)
        sys.exit(1)

    Q = Q[args.skip::max(1, args.stride)]
    if Q.size == 0:
        print("No samples left after skip/stride.", file=sys.stderr)
        sys.exit(1)

    analyze_classical_HO(Q, mass=args.mass, Kspring=args.Kspring, temperature=args.temp)

if __name__ == "__main__":
    main()
