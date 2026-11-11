# VAL_unified.py
# Visualize seed data and PIMC pools in a unified way

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_pool(path, direction=None):
    data = np.load(path, allow_pickle=True)
    if direction in ["forward", "reverse"]:
        try:
            q = data[f"q{'F' if direction == 'forward' else 'R'}"]
            p = data[f"p{'F' if direction == 'forward' else 'R'}"]
            natoms = int(data["natoms"])
            ndim = int(data["ndim"])
            nbeads = int(data["nbeads"])
            meta = data.get("meta", f"seed:{direction}")
            return q, p, natoms, ndim, nbeads, str(meta)
        except KeyError as e:
            raise KeyError(f"Missing key in seed data: {e}")
    elif "q" in data and "p" in data:
        q = data["q"]
        p = data["p"]
        natoms = int(data["natoms"])
        ndim = int(data["ndim"])
        nbeads = int(data["nbeads"])
        meta = data.get("meta", "pimc-pool")
        return q, p, natoms, ndim, nbeads, str(meta)
    else:
        raise KeyError(f"Input file missing required keys for direction={direction}")

def centroid(arr, axis=1):
    return np.mean(arr, axis=axis)

def rp_width(arr):
    return np.std(arr, axis=1)

def is_all_zero(x, atol=0.0):
    return np.all(np.isclose(x, 0.0, atol=atol))

def load_indices(path, nmax):
    arr = np.load(path)
    idx = np.array(arr, dtype=int).ravel()
    return np.unique(idx[(idx >= 0) & (idx < nmax)])

def collect_dataset(path, label, first_coord=False, indices_path=None, direction=None):
    q, p, natoms, ndim, nbeads, meta = load_pool(path, direction=direction)
    qc = centroid(q)
    pc = centroid(p)
    if qc.shape[1] == 1 or first_coord:
        qc_1d = qc[:, 0]; pc_1d = pc[:, 0]
    else:
        qc_1d = qc.mean(axis=1); pc_1d = pc.mean(axis=1)
    zero_p = is_all_zero(pc_1d)
    sel_idx = load_indices(indices_path, nmax=len(qc_1d)) if indices_path else None
    return {
        "label": label,
        "qc": qc_1d,
        "pc": pc_1d,
        "qwidth": rp_width(q).mean(axis=1),
        "zero_p": zero_p,
        "natoms": natoms,
        "ndim": ndim,
        "nbeads": nbeads,
        "meta": meta,
        "sel_idx": sel_idx,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", required=True, help="Pools: .npz or .npz:DIRECTION")
    ap.add_argument("--indices", nargs="*", default=None)
    ap.add_argument("--outdir", default="figs_phase")
    ap.add_argument("--first-coord", action="store_true")
    ap.add_argument("--max-scatter", type=int, default=5000)
    ap.add_argument("--width", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    datasets = []
    for pool_entry in args.pool:
        if ":" in pool_entry:
            path, direction = pool_entry.split(":", 1)
            label = os.path.basename(path).replace(".npz", f":{direction}")
        else:
            path, direction = pool_entry, None
            label = os.path.basename(path).replace(".npz", "")
        idx_path = None
        for ip in args.indices or []:
            if "=" in ip:
                k, v = ip.split("=", 1)
                if k == label:
                    idx_path = v
            elif len(args.pool) == 1:
                idx_path = ip
        datasets.append(collect_dataset(path, label, args.first_coord, idx_path, direction))

    # -- Plots --
    def scatter():
        plt.figure()
        rng = np.random.default_rng(0)
        for ds in datasets:
            qc, pc, sel = ds["qc"], ds["pc"], ds["sel_idx"]
            n = len(qc)
            idx = rng.choice(n, size=min(n, args.max_scatter), replace=False)
            plt.scatter(qc[idx], pc[idx], s=8, alpha=0.3, label=ds["label"])
            if sel is not None:
                plt.scatter(qc[sel], pc[sel], s=22, edgecolors='k', facecolors='none', linewidths=1.0, label=f"{ds['label']} sel")
        plt.xlabel(r"centroid $\bar{q}$"); plt.ylabel(r"centroid $\bar{p}$")
        plt.title("Phase space (centroid)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "phase_scatter.png"), dpi=200); plt.close()

    def hist_q():
        plt.figure()
        for ds in datasets:
            plt.hist(ds["qc"], bins=80, histtype="step", density=True, label=ds["label"])
            if ds["sel_idx"] is not None:
                plt.hist(ds["qc"][ds["sel_idx"]], bins=80, alpha=0.35, density=True, label=ds["label"]+" sel")
        plt.xlabel(r"$\bar{q}$"); plt.ylabel("density")
        plt.title("Centroid position"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "marginal_q.png"), dpi=200); plt.close()

    def hist_p():
        plt.figure()
        for ds in datasets:
            if ds["zero_p"]:
                plt.hist([0.0], bins=3, range=(-1e-12, 1e-12), label=ds["label"]+" (p=0)")
            else:
                plt.hist(ds["pc"], bins=80, histtype="step", density=True, label=ds["label"])
            if ds["sel_idx"] is not None:
                plt.hist(ds["pc"][ds["sel_idx"]], bins=80, alpha=0.35, density=True, label=ds["label"]+" sel")
        plt.xlabel(r"$\bar{p}$"); plt.ylabel("density")
        plt.title("Centroid momentum"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "marginal_p.png"), dpi=200); plt.close()

    def hist_width():
        if not args.width: return
        plt.figure()
        for ds in datasets:
            qw = ds["qwidth"]
            plt.hist(qw, bins=80, histtype="step", density=True, label=ds["label"])
            if ds["sel_idx"] is not None:
                plt.hist(qw[ds["sel_idx"]], bins=80, alpha=0.35, density=True, label=ds["label"]+" sel")
        plt.xlabel("RP width"); plt.ylabel("density")
        plt.title("RP width (std over beads)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "width.png"), dpi=200); plt.close()

    scatter(); hist_q(); hist_p(); hist_width()

if __name__ == "__main__":
    main()
