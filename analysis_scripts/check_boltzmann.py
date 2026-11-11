#!/usr/bin/env python3
# Make OVERLAID histograms of raw positions from two EXTXYZ files. Nothing else.

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read as ase_read


def load_positions(path):
    frames = ase_read(path, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    # stack to (n_samples, 3)
    pos = np.vstack([np.asarray(f.positions, dtype=float) for f in frames])
    return pos  # (frames*natoms, 3)


def main():
    ap = argparse.ArgumentParser(description="Overlay histograms of RAW positions from two EXTXYZ files.")
    ap.add_argument("fileA")
    ap.add_argument("fileB")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--out", default="hist_positions_overlay.png")
    args = ap.parse_args()

    A = load_positions(args.fileA)[1000:]
    B = load_positions(args.fileB)[1000:]

    tagA = os.path.splitext(os.path.basename(args.fileA))[0]
    tagB = os.path.splitext(os.path.basename(args.fileB))[0]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axis_names = ["X", "Y", "Z"]

    for i, axname in enumerate(axis_names):
        ax = axs[i]
        lo = min(A[:, i].min(), B[:, i].min())
        hi = max(A[:, i].max(), B[:, i].max())
        edges = np.linspace(lo, hi, args.bins + 1)

        ax.hist(A[:, i], bins=edges, density=True, alpha=0.5, label=tagA)
        ax.hist(B[:, i], bins=edges, density=True, alpha=0.5, label=tagB)

        ax.set_xlabel("Position")
        ax.set_ylabel("Density")
        ax.set_title(f"Axis {axname}")
        ax.legend()

    fig.suptitle(f"Histogram Overlays (raw positions): {tagA} vs {tagB}")
    fig.savefig(args.out, dpi=200)
    # If you want a window to pop up when running locally, uncomment:
    plt.show()


if __name__ == "__main__":
    main()
