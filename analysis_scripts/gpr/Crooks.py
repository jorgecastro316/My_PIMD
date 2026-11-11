import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from sklearn.neighbors import KernelDensity

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 40
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.fancybox"] = False

def load_work(npz_path, direction):
    f = np.load(npz_path)
    keys = f.files
    print(f"[LOAD] Loading {direction}: {npz_path}")

    if direction == "forward":
        if "wF" in keys:
            w = f["wF"]
        elif "w" in keys:
            w = f["w"]
        elif "qF" in keys and "wF" in keys:
            w = f["wF"]
        else:
            raise KeyError(f"No valid forward work key found in {npz_path}")
    elif direction == "reverse":
        if "wR" in keys:
            w = f["wR"]
        elif "w" in keys:
            w = f["w"]
        elif "qR" in keys and "wR" in keys:
            w = f["wR"]
        else:
            raise KeyError(f"No valid reverse work key found in {npz_path}")
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return -w if direction == "reverse" else w

def main():
    parser = argparse.ArgumentParser(description="Plot Crooks work distributions")
    parser.add_argument("--forward", type=str, required=True, help="Forward work .npz file")
    parser.add_argument("--reverse", type=str, required=True, help="Reverse work .npz file")
    parser.add_argument("--name", type=str, default="free_energy_diff", help="Output plot name")
    parser.add_argument("--max-points", type=int, default=10**6, help="Max number of points to plot")
    args = parser.parse_args()

    WAtoB = load_work(args.forward, "forward")
    WBtoA = load_work(args.reverse, "reverse")

    N = min(len(WAtoB), len(WBtoA), args.max_points)
    WAtoB = WAtoB[:N]
    WBtoA = WBtoA[:N]
    print(f"[INFO] Using {N} work values from each direction")

    # KDE fits
    grid = np.linspace(-6, 6, 1000)[:, np.newaxis]
    kdef = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(WAtoB.reshape(-1, 1))
    kdeb = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(WBtoA.reshape(-1, 1))

    log_densf = kdef.score_samples(grid)
    log_densb = kdeb.score_samples(grid)

    densf = np.exp(log_densf)
    densb = np.exp(log_densb)

    # Find intersection points
    intersection_idx = np.argwhere(np.diff(np.sign(densf - densb)) != 0).reshape(-1)
    intersection_points = grid[intersection_idx]
    intersection_y_values = (densf[intersection_idx] + densb[intersection_idx]) / 2

    if len(intersection_points) == 0:
        print("[WARN] No intersection point found")
        highest_intersection_x = np.nan
    else:
        highest_intersection_idx = np.argmax(intersection_y_values)
        highest_intersection_x = intersection_points[highest_intersection_idx]
        print('Free energy Difference (ΔF):', np.around(highest_intersection_x, 6), 'Hartree')

    # Save KDE data
    np.savetxt("work_forward.txt", np.column_stack([grid.flatten(), densf]))
    np.savetxt("work_backward.txt", np.column_stack([grid.flatten(), densb]))

    # Plot
    fig, ax = plt.subplots(dpi=100)
    fig.set_size_inches(12, 8)
    ax.plot(grid, densf, label='Forward', linewidth=3.5, color='r')
    ax.plot(grid, densb, label='Backward', linewidth=3.5, color='b')
    ax.set_xlabel('Work [Hartree]')
    ax.set_ylabel('Distribution')
    ax.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig(args.name + '.png', format='png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
