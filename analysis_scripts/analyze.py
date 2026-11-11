import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_columns.py <filename> <col1> [<col2> ...] "
              "[--skip N] [--hist] [--out output.png]")
        sys.exit(1)

    args = sys.argv[1:]

    # Handle optional flags
    skip_lines = 0
    make_hist = False
    output_name = None

    if '--skip' in args:
        skip_idx = args.index('--skip')
        try:
            skip_lines = int(args[skip_idx + 1])
        except (IndexError, ValueError):
            print("Error: '--skip' must be followed by an integer.")
            sys.exit(1)
        del args[skip_idx:skip_idx + 2]

    if '--hist' in args:
        make_hist = True
        args.remove('--hist')

    if '--out' in args:
        out_idx = args.index('--out')
        try:
            output_name = args[out_idx + 1]
        except IndexError:
            print("Error: '--out' must be followed by a filename.")
            sys.exit(1)
        del args[out_idx:out_idx + 2]

    # Parse filename and column indices
    filename = args[0]
    try:
        column_indices = [int(i) for i in args[1:]]
    except ValueError:
        print("Error: column indices must be integers.")
        sys.exit(1)

    # Load data
    try:
        data = np.loadtxt(filename, skiprows=skip_lines, comments='#')
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    time = data[:, 0]

    for col in column_indices:
        if col >= data.shape[1]:
            print(f"Column index {col} is out of bounds for file with {data.shape[1]} columns.")
            continue

        values = data[:, col]
        avg = np.mean(values)
        std = np.std(values)

        print(f"Column {col}: mean = {avg:.6f}, std = {std:.6f}")

        if make_hist:
            plt.hist(values, bins=50, alpha=0.7, label=f"Column {col}")
        else:
            plt.plot(time, values, label=f"Column {col}")

    if make_hist:
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Selected Columns")
    else:
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Selected Columns Over Time")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_name:
        plt.savefig(output_name, dpi=300)
        print(f"Figure saved as {output_name}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
