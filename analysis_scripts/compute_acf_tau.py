import sys
import numpy as np
import matplotlib.pyplot as plt

def autocorrelation(x):
    x = np.asarray(x)
    x -= np.mean(x)
    n = len(x)
    acf = np.correlate(x, x, mode='full')[n - 1:]
    acf /= np.var(x) * np.arange(n, 0, -1)
    return acf

def integrated_autocorrelation_time(acf):
    tau = 0.5  # initial value includes ACF[0]
    for t in range(1, len(acf)):
        if acf[t] < 0:
            break
        tau += acf[t]
    return tau

def main():
    if len(sys.argv) < 3:
        print("Usage: python compute_acf_tau.py <filename> <column> [--skip N] [--dt fs] [--stride N] [--plot]")
        sys.exit(1)

    filename = sys.argv[1]
    col = int(sys.argv[2])
    skip = 0
    dt = 1.0
    stride = 1
    plot = False

    args = sys.argv[3:]
    if '--skip' in args:
        skip = int(args[args.index('--skip') + 1])
    if '--dt' in args:
        dt = float(args[args.index('--dt') + 1])
    if '--stride' in args:
        stride = int(args[args.index('--stride') + 1])
    if '--plot' in args:
        plot = True

    try:
        data = np.loadtxt(filename, skiprows=skip, comments='#')
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    if col >= data.shape[1]:
        print(f"Error: file only has {data.shape[1]} columns.")
        sys.exit(1)

    values = data[:, col]
    acf = autocorrelation(values)
    tau_samples = integrated_autocorrelation_time(acf)
    tau_fs = tau_samples * stride * dt
    tau_steps = tau_fs / dt

    print(f"Integrated autocorrelation time:")
    print(f"- In samples          : {tau_samples:.2f}")
    print(f"- In femtoseconds     : {tau_fs:.2f} fs")
    print(f"- In simulation steps : {round(tau_steps)} steps")

    if plot:
        time_axis = np.arange(len(acf)) * stride * dt
        plt.plot(time_axis, acf)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel("Lag time [fs]")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation Function")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
