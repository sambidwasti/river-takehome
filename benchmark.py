"""Part 3 benchmarks: time a simulator version on brickwork circuits.

Two regimes, as the challenge asks:

* qubits: noiseless, 1 shot, depth 20, n from --n-min upwards. Records
  wall-clock time and peak memory for each n. Stops after --n-max, or
  once a run takes longer than --max-seconds (each extra qubit roughly
  doubles the time, so the next n would take about twice as long).
* shots: n = 12, depth 20, p_gate = p_idle = 1e-3, p_readout = 1e-2.
  Records shots per second.

--impl picks the simulator version:
  reference -> sim_reference.py (copy of sim.py at tag v1-reference)
  current   -> sim.py (the version being optimised)

Every n (and the shots run) runs in its own Python process, so the peak
memory reported is for that run alone. Only the simulation is timed:
building and parsing the circuit happen before the clock starts.

Results are appended to a CSV file with the date and time, the git
commit and the settings, so every number can be traced to the code
that produced it.

Examples::

    python3 benchmark.py qubits --impl reference --n-max 18
    python3 benchmark.py shots --impl reference --shots 200
"""

import argparse
import csv
import datetime
import importlib
import json
import os
import resource
import subprocess
import sys
import time

from brickwork import brickwork_circuit
from read_circuit import parse_circuit

IMPLS = {"reference": "sim_reference", "current": "sim"}
CSV_FIELDS = ["timestamp", "commit", "impl", "regime", "n_qubits", "depth", "shots",
              "p_gate", "p_idle", "p_readout", "seconds", "shots_per_second",
              "peak_memory_mb"]


# ---- worker: one run, in its own process ----

def peak_memory_mb():
    """Peak resident memory of this process, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def run_once(impl, n_qubits, depth, shots, p_gate, p_idle, p_readout, seed):
    """Build the circuit, then time sample_counts. Returns a result dict."""
    sim = importlib.import_module(IMPLS[impl])
    n, steps = parse_circuit(brickwork_circuit(n_qubits, depth, seed))

    start = time.perf_counter()
    sim.sample_counts(n, steps, shots, p_gate, p_idle, p_readout, seed)
    seconds = time.perf_counter() - start

    return {"seconds": seconds, "shots_per_second": shots / seconds,
            "peak_memory_mb": peak_memory_mb()}


def run_in_subprocess(impl, n_qubits, depth, shots, p_gate, p_idle, p_readout, seed):
    """Run run_once in a fresh Python process and return its result dict."""
    cmd = [sys.executable, os.path.abspath(__file__), "worker",
           "--impl", impl, "--n", str(n_qubits), "--depth", str(depth),
           "--shots", str(shots), "--p-gate", str(p_gate), "--p-idle", str(p_idle),
           "--p-readout", str(p_readout), "--seed", str(seed)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__)))
    return json.loads(out.stdout)


# ---- recording ----

def git_commit():
    """Short commit hash, with '-dirty' if there are uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], check=True,
                               capture_output=True, text=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_row(path, row):
    """Append one result row to the CSV file, writing the header if it is new."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


# ---- regimes ----

def bench_qubits(args, stamp, commit):
    print(f"{'n':>3}  {'seconds':>10}  {'peak MB':>9}")
    for n in range(args.n_min, args.n_max + 1):
        r = run_in_subprocess(args.impl, n, args.depth, 1, 0.0, 0.0, 0.0, args.seed)
        print(f"{n:3d}  {r['seconds']:10.4f}  {r['peak_memory_mb']:9.1f}", flush=True)
        append_row(args.out, {
            "timestamp": stamp, "commit": commit, "impl": args.impl, "regime": "qubits",
            "n_qubits": n, "depth": args.depth, "shots": 1,
            "p_gate": 0.0, "p_idle": 0.0, "p_readout": 0.0, **r})
        if r["seconds"] > args.max_seconds:
            print(f"stopping: {r['seconds']:.1f} s > --max-seconds {args.max_seconds}")
            break


def bench_shots(args, stamp, commit):
    r = run_in_subprocess(args.impl, args.n, args.depth, args.shots,
                          args.p_gate, args.p_idle, args.p_readout, args.seed)
    print(f"n = {args.n}, depth = {args.depth}, shots = {args.shots}: "
          f"{r['seconds']:.2f} s, {r['shots_per_second']:.1f} shots/s, "
          f"peak {r['peak_memory_mb']:.1f} MB")
    append_row(args.out, {
        "timestamp": stamp, "commit": commit, "impl": args.impl, "regime": "shots",
        "n_qubits": args.n, "depth": args.depth, "shots": args.shots,
        "p_gate": args.p_gate, "p_idle": args.p_idle, "p_readout": args.p_readout, **r})


# ---- command line ----

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a simulator version.")
    sub = parser.add_subparsers(dest="regime", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--impl", choices=sorted(IMPLS), default="current")
    common.add_argument("--depth", type=int, default=20)
    common.add_argument("--seed", type=int, default=1)
    common.add_argument("--out", default="results/benchmarks.csv")

    q = sub.add_parser("qubits", parents=[common], help="noiseless, 1 shot, n increasing")
    q.add_argument("--n-min", type=int, default=4)
    q.add_argument("--n-max", type=int, default=30)
    q.add_argument("--max-seconds", type=float, default=60.0)

    s = sub.add_parser("shots", parents=[common], help="n = 12 with noise, many shots")
    s.add_argument("--n", type=int, default=12)
    s.add_argument("--shots", type=int, default=100_000)
    s.add_argument("--p-gate", type=float, default=1e-3)
    s.add_argument("--p-idle", type=float, default=1e-3)
    s.add_argument("--p-readout", type=float, default=1e-2)

    w = sub.add_parser("worker", parents=[common], help=argparse.SUPPRESS)
    w.add_argument("--n", type=int, required=True)
    w.add_argument("--shots", type=int, required=True)
    w.add_argument("--p-gate", type=float, required=True)
    w.add_argument("--p-idle", type=float, required=True)
    w.add_argument("--p-readout", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.regime == "worker":
        print(json.dumps(run_once(args.impl, args.n, args.depth, args.shots,
                                  args.p_gate, args.p_idle, args.p_readout, args.seed)))
        return

    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    commit = git_commit()
    print(f"impl = {args.impl} ({IMPLS[args.impl]}.py), commit = {commit}, {stamp}")
    if args.regime == "qubits":
        bench_qubits(args, stamp, commit)
    else:
        bench_shots(args, stamp, commit)
    print("appended to", args.out)


if __name__ == "__main__":
    main()
