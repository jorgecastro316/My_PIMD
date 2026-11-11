#!/usr/bin/env python3
# compare_pimc_phase_space.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_pool(path):
    data = np.load(path, allow_pickle=True)
    q = data["q"]            # (n_samples, nbeads, natoms*ndim)
    p = data["p"]
    natoms = int(data["natoms"])
    ndim   = int(data["ndim"])
    nbeads = int(data["nbeads"])
    meta   = data.get("meta", "")
    if hasattr(meta, "item"):  # handle 0d np.array
        try: meta = meta.item()
        except: meta = str(meta)
    return q, p, natoms, ndim, nbeads, str(meta)

def centroid(arr, axis=1):
    return np.mean(arr, axis=axis)  # over beads

def rp_width(arr):
    return np.std(arr, axis=1)      # std over beads per sample

def is_all_zero(x, atol=0.0):
    return np.all(np.isclose(x, 0.0, atol=atol))

def parse_indices_args(indices_args, single_label=None):
    """
    Parse --indices CLI:
      - single pool: ["path.npy"]  -> {single_label: path.npy}
      - multi pools: ["L1=path.npy", "L2=path.npy"] -> {L1: path, L2: path}
    """
    if not indices_args:
        return {}
    mapping = {}
    # single pool: just provide the file
    if single_label and len(indices_args) == 1 and "=" not in indices_args[0]:
        mapping[single_label] = indices_args[0]
        return mapping

    # multi or explicit mapping
    for item in indices_args:
        if "=" not in item:
            raise SystemExit("For multiple pools, use --indices label=idx.npy ...")
        lab, path = item.split("=", 1)
        mapping[lab] = path
    return mapping

def load_indices(path, nmax):
    arr = np.load(path)
    idx = np.array(arr, dtype=int).ravel()
    # keep only in-range unique sorted indices
    idx = np.unique(idx[(idx >= 0) & (idx < nmax)])
    return idx

def collect_dataset(path, label, first_coord, indices_path=None):
    q, p, natoms, ndim, nbeads, meta = load_pool(path)
    qc = centroid(q)   # (n_samples, natoms*ndim)
    pc = centroid(p)
    if qc.shape[1] == 1 or first_coord:
        qc_1d = qc[:, 0]
        pc_1d = pc[:, 0]
    else:
        # if multiple coords: average across coords to 1D summary
        qc_1d = qc.mean(axis=1)
        pc_1d = pc.mean(axis=1)
    zero_p = is_all_zero(pc_1d)

    sel_idx = None
    if indices_path is not None:
        sel_idx = load_indices(indices_path, nmax=qc_1d.shape[0])

    return {
        "label": label,
        "qc": qc_1d,
        "pc": pc_1d,
        "qwidth": rp_width(q).mean(axis=1),  # width averaged across coords
        "zero_p": zero_p,
        "natoms": natoms,
        "ndim": ndim,
        "nbeads": nbeads,
        "meta": meta,
        "sel_idx": sel_idx,  # np.ndarray[int] or None
    }

def main():
    ap = argparse.ArgumentParser(description="Compare/inspect PIMC phase space (1D centroid).")
    ap.add_argument("--pool", nargs="+", required=True,
                    help="Single: PATH; Multiple: PATH1 LABEL1 PATH2 LABEL2 ...")
    ap.add_argument("--indices", nargs="*", default=None,
                    help="Optional selection indices. Single pool: path.npy. "
                         "Multiple pools: LABEL=path.npy (repeatable).")
    ap.add_argument("--outdir", default="figs_phase", help="Directory to save plots")
    ap.add_argument("--max-scatter", type=int, default=5000,
                    help="Max points per method in scatter for readability")
    ap.add_argument("--first-coord", action="store_true",
                    help="If natoms*ndim > 1, use only the first coordinate for plotting")
    ap.add_argument("--width", action="store_true",
                    help="Also plot ring-polymer width (std over beads)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    datasets = []
    indices_map = {}
    single_label = None

    if len(args.pool) == 1:
        path = args.pool[0]
        single_label = os.path.splitext(os.path.basename(path))[0]
        indices_map = parse_indices_args(args.indices, single_label=single_label)
        idx_path = indices_map.get(single_label, None)
        datasets.append(collect_dataset(path, single_label, args.first_coord, indices_path=idx_path))
    else:
        if len(args.pool) % 2 != 0:
            raise SystemExit("Error: when passing multiple pools, use pairs PATH LABEL ...")
        # build mapping for multiple labels
        indices_map = parse_indices_args(args.indices, single_label=None)
        for i in range(0, len(args.pool), 2):
            path, label = args.pool[i], args.pool[i+1]
            idx_path = indices_map.get(label, None)
            datasets.append(collect_dataset(path, label, args.first_coord, indices_path=idx_path))

    # -------- Phase-space scatter (q̄ vs p̄) --------
    plt.figure()
    rng = np.random.default_rng(0)
    for ds in datasets:
        qc, pc, sel = ds["qc"], ds["pc"], ds["sel_idx"]
        n = len(qc)
        plot_qc, plot_pc = qc, pc
        if args.max_scatter and n > args.max_scatter:
            idx = rng.choice(n, size=args.max_scatter, replace=False)
            plot_qc = qc[idx]; plot_pc = pc[idx]
            # If we sampled for display, still overlay ALL selected (distinct) below
        lbl = ds["label"] + (" (p=0)" if ds["zero_p"] else "")
        # background cloud
        plt.scatter(plot_qc, plot_pc, s=8, alpha=0.35, label=lbl)
        # overlay selected indices (if provided)
        if sel is not None and sel.size > 0:
            plt.scatter(qc[sel], pc[sel], s=22, alpha=0.95,
                        facecolors='none', edgecolors='k', linewidths=1.0,
                        label=f"{ds['label']} (selected {len(sel)})")
    plt.xlabel(r"centroid position $\bar{q}$")
    plt.ylabel(r"centroid momentum $\bar{p}$")
    plt.title("PIMC phase space (centroid) with selected overlays")
    plt.legend(markerscale=1.5)
    plt.tight_layout()
    fig1 = os.path.join(args.outdir, "phase_scatter.png")
    plt.savefig(fig1, dpi=200); plt.close()
    print(f"[SAVE] {fig1}")

    # -------- Marginal histogram: q̄ --------
    plt.figure()
    for ds in datasets:
        qc, sel = ds["qc"], ds["sel_idx"]
        # full
        plt.hist(qc, bins=80, histtype="step", density=True, label=ds["label"])
        # selected overlay
        if sel is not None and sel.size > 0:
            plt.hist(qc[sel], bins=80, density=True, alpha=0.35,
                     label=f"{ds['label']} sel")
    plt.xlabel(r"centroid position $\bar{q}$"); plt.ylabel("density")
    plt.title("Marginal of centroid position (full vs selected)")
    plt.legend(); plt.tight_layout()
    fig2 = os.path.join(args.outdir, "marginal_q.png")
    plt.savefig(fig2, dpi=200); plt.close()
    print(f"[SAVE] {fig2}")

    # -------- Marginal histogram: p̄ --------
    plt.figure()
    for ds in datasets:
        pc, sel = ds["pc"], ds["sel_idx"]
        if ds["zero_p"]:
            # Full distribution is degenerate; just mark zeros
            plt.hist([0.0], bins=3, range=(-1e-12, 1e-12),
                     histtype="step", density=False, label=ds["label"]+" (zeros)")
            if sel is not None and sel.size > 0:
                plt.hist([0.0], bins=3, range=(-1e-12, 1e-12),
                         alpha=0.35, label=ds["label"]+" sel")
        else:
            plt.hist(pc, bins=80, histtype="step", density=True, label=ds["label"])
            if sel is not None and sel.size > 0:
                plt.hist(pc[sel], bins=80, density=True, alpha=0.35, label=ds["label"]+" sel")
    plt.xlabel(r"centroid momentum $\bar{p}$"); plt.ylabel("density")
    plt.title("Marginal of centroid momentum (full vs selected)")
    plt.legend(); plt.tight_layout()
    fig3 = os.path.join(args.outdir, "marginal_p.png")
    plt.savefig(fig3, dpi=200); plt.close()
    print(f"[SAVE] {fig3}")

    # -------- Optional: ring-polymer width --------
    if args.width:
        plt.figure()
        for ds in datasets:
            qw, sel = ds["qwidth"], ds["sel_idx"]
            plt.hist(qw, bins=80, histtype="step", density=True, label=ds["label"])
            if sel is not None and sel.size > 0:
                plt.hist(qw[sel], bins=80, density=True, alpha=0.35, label=ds["label"]+" sel")
        plt.xlabel("ring-polymer width (std over beads, averaged over coords)")
        plt.ylabel("density")
        plt.title("Distribution of ring-polymer width (full vs selected)")
        plt.legend(); plt.tight_layout()
        fig4 = os.path.join(args.outdir, "width.png")
        plt.savefig(fig4, dpi=200); plt.close()
        print(f"[SAVE] {fig4}")

if __name__ == "__main__":
    main()
