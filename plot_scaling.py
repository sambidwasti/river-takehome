"""Many-qubits regime: plot wall-clock time against n (log scale), one line per version.

Reads results/benchmarks.csv (noiseless, 1 shot, D = 20 rows). Where a version was run more
than once at the same n (the reference was run twice for n = 4-19), the plot uses the mean
of the runs. Extension runs of a version are joined to its original sweep.

Run with::

    python3 plot_scaling.py
"""

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/benchmarks.csv"
OUT = "figures/qubits_scaling.png"

#: Commit in the CSV -> version label. Old rows carry pre-rewrite hashes; extension rows the
#: new ones. The reference is identified by impl = "reference".
VERSIONS = {
    "Optimisation 1: vectorise apply_1q": {"f8ab0c1-dirty", "b993111"},
    "Optimisation 2: vectorise apply_2q": {"93eba8b-dirty", "c422a37"},
    "Optimisation 3b: CNOT as a swap": {"7e31641-dirty", "c667dbe"},
    "Optimisation 3c+: in place, X/Y/Z kernels": {"7991f4f"},
    "Optimisation 4: numba apply_1q (final)": {"38cd325-dirty"},
}
REFERENCE = "Reference (Part 1)"


def load():
    """Return {label: {n: [seconds, ...]}} for the qubits regime."""
    data = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(open(CSV)):
        if row["regime"] != "qubits":
            continue
        if row["impl"] == "reference":
            label = REFERENCE
        else:
            label = next((k for k, v in VERSIONS.items() if row["commit"] in v), None)
            if label is None:
                continue  # reverted or superseded rounds are not plotted
        data[label][int(row["n_qubits"])].append(float(row["seconds"]))
    return data


def main():
    data = load()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in [REFERENCE, *VERSIONS]:
        ns = sorted(data[label])
        times = [np.mean(data[label][n]) for n in ns]
        ax.plot(ns, times, "o-", markersize=4, label=f"{label} (n ≤ {ns[-1]})")
    ax.set_yscale("log")
    ax.set_xlabel("number of qubits n")
    ax.set_ylabel("wall-clock time (s), log scale")
    ax.set_title("Many qubits: noiseless brickwork circuit, D = 20, 1 shot")
    ax.set_xticks(range(4, 30, 2))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)
    for label in [REFERENCE, *VERSIONS]:
        ns = sorted(data[label])
        print(f"{label:<45} n = {ns[0]}–{ns[-1]}, runs per n: "
              f"{sorted({len(data[label][n]) for n in ns})}")


if __name__ == "__main__":
    main()
