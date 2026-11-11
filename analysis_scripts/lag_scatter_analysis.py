import argparse
import matplotlib.pyplot as plt

def read_1d_positions(filename):
    """Extracts 1D positions from a trajectory file."""
    positions = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            line = lines[i + 2]
            if line.startswith("H"):
                positions.append(float(line.split()[1]))
    return positions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Trajectory file with 1D positions")
    parser.add_argument("--lag", type=int, default=1, help="Lag in number of steps")
    args = parser.parse_args()

    data = read_1d_positions(args.filename)
    lag = args.lag

    if len(data) <= lag:
        print("Error: Trajectory too short for the requested lag.")
        return

    x_t = data[:-lag]
    x_t_lag = data[lag:]

    plt.figure(figsize=(6, 5))
    plt.scatter(x_t, x_t_lag, alpha=0.5)
    plt.xlabel(f"x(t)")
    plt.ylabel(f"x(t + {lag})")
    plt.title(f"Lagged Scatter Plot (lag = {lag})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
