# Noisy quantum circuit simulator

A state-vector simulator for noisy quantum circuits, written in Python with NumPy and numba.
The design, results and figures are in [WRITEUP.md](WRITEUP.md) (about 2 pages), with the
full version in [WRITEUP_DETAILED.md](WRITEUP_DETAILED.md); the round-by-round record of
the optimisation work is in [results/progress.md](results/progress.md).

## Setup

Python 3.12 (tested on 3.12.2, macOS, Apple M3 Pro; see [hardware.md](hardware.md)).

```bash
python3 -m pip install -r requirements.txt
```

## Run the simulator

```bash
python3 sim.py ghz.txt --shots 1000 --p-gate 0.001 --p-idle 0.001 --p-readout 0.01 --seed 1
```

It prints the measured counts, as bitstrings with qubit 0 rightmost.
All noise parameters default to 0, so `python3 sim.py ghz.txt --shots 1000` runs noiselessly.

### Circuit format

```
QUBITS 3
# prepare a 3-qubit GHZ state
H 0
TICK
CNOT 0 1
TICK
CNOT 1 2
TICK
RZ 0.785 0
```

- First line: `QUBITS n`. Then one gate per line: name, angle in radians (RX, RY, RZ only), then
  the qubits, control first.
- Gates: X, Y, Z, H, S, T, RX, RY, RZ, CNOT, CZ.
- `TICK` ends a time step; qubits without a gate in a step are idle. Consecutive TICKs give an
  all-idle step; the end of the file closes the last step. Lines starting with `#` are comments.

## Tests

```bash
python3 -m pytest
```

| File | What it checks |
|---|---|
| `test_gates.py` | each gate on a simple input gives what its matrix predicts |
| `test_noise.py` | zero noise gives the ideal result; readout, gate and idle noise flip at the expected rates |
| `test_invariants.py` | the state stays normalised; a gate followed by its inverse returns the original state |
| `test_equivalence.py` | the optimised `sim.py` matches the reference `sim_reference.py`: exact states without noise, identical counts with the same seed, and a chi-squared test with different seeds |

## Part 2: idle-noise experiment

```bash
python3 idle_noise.py --p-idle 0.05 --p-readout 0.02 --shots 10000 --seed 1
```

Writes `figures/idle_decay.png` (data and fit) and `figures/idle_decay.txt` (table of m,
measured fraction, error and model value), and prints the fitted p_idle.

## Part 3: benchmarks

Generate a benchmark circuit (n qubits, depth D):

```bash
python3 brickwork.py 12 20 --seed 1 > brickwork_n12_d20.txt
```

Many qubits (noiseless, 1 shot, D = 20, n from 4 until a run exceeds `--max-seconds`, default 300 s):

```bash
python3 benchmark.py qubits --impl current
```

Many shots (n = 12, D = 20, p_gate = p_idle = 10⁻³, p_readout = 10⁻², 10⁵ shots by default):

```bash
python3 benchmark.py shots --impl current
```

- `--impl current` runs `sim.py`; `--impl reference` runs the Part 1 reference
  (`sim_reference.py`, git tag `v1-reference`).
- Each size runs in a fresh subprocess; time and peak memory are printed and appended to
  `results/benchmarks.csv` with a timestamp and the git commit.
- Note: the reference is slow (about 2.6 shots/s; n = 21 takes about 6 minutes).

Plot time against n (log scale) from `results/benchmarks.csv`:

```bash
python3 plot_scaling.py
```

Profile one run (example: many-qubits regime at n = 20):

```bash
python3 -m cProfile -s tottime benchmark.py worker --impl current --depth 20 --seed 1 --n 20 --shots 1 --p-gate 0 --p-idle 0 --p-readout 0
```

## Files

| Path | Contents |
|---|---|
| `sim.py` | the simulator (optimised): gate kernels, noise, sampling, command line |
| `sim_reference.py` | the Part 1 reference simulator, unchanged |
| `read_circuit.py` | circuit-file parser |
| `idle_noise.py` | Part 2 experiment: sweep, fit and plot |
| `brickwork.py` | Part 3 benchmark-circuit generator |
| `benchmark.py` | Part 3 benchmark runner (both regimes) |
| `plot_scaling.py` | many-qubits plot from `results/benchmarks.csv` |
| `test_*.py` | tests (see above) |
| `ghz.txt`, `one_gate.txt`, `parse_check.txt`, `ghz_invalid_parse.txt` | example and test circuits |
| `figures/` | plots and the Part 2 table |
| `results/` | benchmark logs, cProfile outputs and `progress.md` (the optimisation record) |
| `hardware.md` | the machine the benchmarks ran on |
| `WRITEUP.md` | the write-up (about 2 pages) |
| `WRITEUP_DETAILED.md` | the full write-up: every round, measurement and derivation |
