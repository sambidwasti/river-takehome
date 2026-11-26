"""Part 2: idle-noise experiment.

Put the qubit in 1 with X, leave it idle for m steps, then measure.
Repeat for a range of m, fit the expected curve and compare the fitted
p_idle with the injected one. Uses the Part 1 simulator unchanged.

Run with::

    python3 idle_noise.py --p-idle 0.05 --p-readout 0.02 --shots 10000 --seed 1
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from read_circuit import parse_circuit
from sim import sample_counts


def idle_circuit(m):
    """Return circuit text: X on qubit 0, then m idle steps.

    The first TICK closes the X step; each TICK after it is one empty step.

    >>> print(idle_circuit(2), end="")
    QUBITS 1
    X 0
    TICK
    TICK
    TICK
    """
    return "QUBITS 1\nX 0\n" + "TICK\n" * (m + 1)


def idle_run(m, shots, p_idle, p_readout, seed):
    """Run the idle circuit for m steps (no gate noise).

    Returns (f, err): the share of shots that read 1, and its error bar
    sqrt(f * (1 - f) / shots).
    """
    n_qubits, steps = parse_circuit(idle_circuit(m))
    counts = sample_counts(n_qubits, steps, shots, 0.0, p_idle, p_readout, seed)
    f = counts.get("1", 0) / shots
    err = math.sqrt(f * (1 - f) / shots)
    return f, err


def model(m, p_idle, p_readout):
    """Expected share of 1s after m idle steps.

    Each idle step flips the bit with probability q = 2/3 * p_idle (X or Y
    flips it, Z does not); readout flips it with probability p_readout:
    f(m) = 1/2 + 1/2 * (1 - 2 * p_readout) * (1 - 2 * q) ** m
    """
    q = 2 * p_idle / 3
    return 0.5 + 0.5 * (1 - 2 * p_readout) * (1 - 2 * q) ** m


def sweep(ms, shots, p_idle, p_readout, seed):
    """Run idle_run for every m. Returns arrays (fs, errs).

    Each m gets its own seed (seed + i), so the points are independent.
    """
    fs, errs = [], []
    for i, m in enumerate(ms):
        run_seed = None if seed is None else seed + i
        f, err = idle_run(m, shots, p_idle, p_readout, run_seed)
        fs.append(f)
        errs.append(err)
    return np.array(fs), np.array(errs)


def fit_p_idle(ms, fs, errs, p_readout):
    """Fit p_idle with p_readout treated as known. Returns (p_fit, p_err)."""
    popt, pcov = curve_fit(
        lambda m, p: model(m, p, p_readout),
        ms, fs,
        p0=[0.01],
        sigma=errs,
        absolute_sigma=True,   # errs are real error sizes
        bounds=(0, 0.75),      # above 0.75, 1 - 2q goes negative
    )
    return popt[0], math.sqrt(pcov[0][0])


def plot_decay(ms, fs, errs, p_fit, p_err, p_idle, p_readout, shots, path):
    """Plot the data with error bars and the fitted curve; save to path."""
    m_fine = np.linspace(0, ms.max(), 200)
    fig, ax = plt.subplots()
    ax.errorbar(ms, fs, yerr=errs, fmt="o", capsize=3,
                label=f"simulated ({shots} shots per point)")
    ax.plot(m_fine, model(m_fine, p_fit, p_readout),
            label=f"fit: p_idle = {p_fit:.4f} ± {p_err:.4f}")
    ax.set_xlim(left=0)   # start the axis exactly at m = 0
    ax.set_xlabel("idle steps m")
    ax.set_ylabel("fraction of shots reading 1")
    ax.set_title(f"Idle decay (injected p_idle = {p_idle}, p_readout = {p_readout})")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2: idle-noise experiment.")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--p-idle", type=float, default=0.05)
    parser.add_argument("--p-readout", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--m-max", type=int, default=60)
    parser.add_argument("--m-step", type=int, default=5)
    parser.add_argument("--out", default="figures/idle_decay.png")
    parser.add_argument("--table", default="figures/idle_decay.txt")
    return parser.parse_args()


def write_table(ms, fs, errs, p_idle, p_readout, path):
    """Save m, measured f, its error and the formula value (injected p) to a text file."""
    with open(path, "w") as out:
        out.write(f"# injected p_idle = {p_idle}, p_readout = {p_readout}\n")
        out.write(f"{'m':>4}  {'f_measured':>10}  {'err':>8}  {'f_formula':>9}\n")
        for m, f, err in zip(ms, fs, errs):
            out.write(f"{m:4d}  {f:10.4f}  {err:8.4f}  "
                      f"{model(m, p_idle, p_readout):9.4f}\n")


def main():
    args = parse_args()
    ms = np.arange(0, args.m_max + 1, args.m_step)

    fs, errs = sweep(ms, args.shots, args.p_idle, args.p_readout, args.seed)
    p_fit, p_err = fit_p_idle(ms, fs, errs, args.p_readout)

    print(" m      f       err")
    for m, f, err in zip(ms, fs, errs):
        print(f"{m:3d}  {f:.4f}  {err:.4f}")
    print(f"injected p_idle = {args.p_idle}")
    print(f"fitted   p_idle = {p_fit:.5f} ± {p_err:.5f}")
    print(f"difference      = {(p_fit - args.p_idle) / p_err:+.2f} sigma")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_decay(ms, fs, errs, p_fit, p_err, args.p_idle, args.p_readout,
               args.shots, args.out)
    print("saved", args.out)

    os.makedirs(os.path.dirname(args.table) or ".", exist_ok=True)
    write_table(ms, fs, errs, args.p_idle, args.p_readout, args.table)
    print("saved", args.table)


if __name__ == "__main__":
    main()
