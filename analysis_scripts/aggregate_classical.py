#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

COLUMNS = ["time", "total", "kinetic", "potential", "temperature"]


def parse_file(path):
    data = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < len(COLUMNS):
                continue
            try:
                vals = [float(parts[i]) for i in range(len(COLUMNS))]
            except ValueError:
                continue
            data.append(vals)
    return np.array(data, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate simulation.classical.*.out files into one NPZ"
    )
    parser.add_argument(
        "root",
        help="Root directory containing run subdirectories with simulation.classical.*.out",
    )
    parser.add_argument(
        "--out",
        default="aggregated_classical.npz",
        help="Output NPZ path (default: aggregated_classical.npz)",
    )
    parser.add_argument(
        "--hist-column",
        help="Optional column name or 0-based index to plot histogram",
    )
    parser.add_argument("--bins", type=int, default=100, help="Histogram bins")
    args = parser.parse_args()

    master = {c: [] for c in COLUMNS}
    nfiles = 0
    for dirpath, _, filenames in os.walk(args.root):
        for name in filenames:
            if name.startswith("simulation.classical") and name.endswith(".out"):
                path = os.path.join(dirpath, name)
                arr = parse_file(path)
                if arr.size == 0:
                    continue
                nfiles += 1
                for idx, col in enumerate(COLUMNS):
                    master[col].append(arr[:, idx])
    if nfiles == 0:
        raise SystemExit(
            f"No simulation.classical.*.out files found under {args.root}"
        )

    merged = {}
    for col in COLUMNS:
        merged[col] = np.concatenate(master[col])
        print(
            f"{col:>10s}: mean={merged[col].mean():.6e} std={merged[col].std():.6e}"
            f" samples={merged[col].size}"
        )
    np.savez(args.out, **merged)
    print(f"[SAVE] wrote {args.out}")

    if args.hist_column:
        if args.hist_column.isdigit():
            idx = int(args.hist_column)
            if idx < 0 or idx >= len(COLUMNS):
                raise SystemExit(f"Column index {idx} out of range for {COLUMNS}")
            col = COLUMNS[idx]
        else:
            if args.hist_column not in COLUMNS:
                raise SystemExit(f"Unknown column '{args.hist_column}'. Choose from {COLUMNS} or indices 0-{len(COLUMNS)-1}.")
            col = args.hist_column
        data = merged[col]
        plt.figure()
        plt.hist(data, bins=args.bins, density=True)
        plt.xlabel(col)
        plt.ylabel("Probability density")
        plt.tight_layout()
        hist_path = os.path.splitext(args.out)[0] + f"_{col}_hist.png"
        plt.savefig(hist_path, dpi=200)
        plt.close()
        print(f"[SAVE] wrote {hist_path}")


if __name__ == "__main__":
    main()
