# Quantum circuit simulator: write-up
11/26/26 
Sambid Wasti

**Approach.** Build the simplest simulator I could trust, pin it down with tests, validate the
noise end to end, then make it fast in measured, recorded steps (Part 3, Method).

## Part 1: a correct simulator

The simulator follows the state-vector model, gate set, noise model and circuit format in the
brief. Two design decisions:

- **Clarity first.** Gates are matrices applied by two plain functions, `apply_1q` and
  `apply_2q`, written as direct loops over the amplitudes. Slow, but easy to check against the
  definitions, which made it a trustworthy reference (tagged `v1-reference`) for Part 3.
- **Separate parsing from simulation.** `read_circuit.py` turns the circuit file into a list of
  time steps; `sim.py` only runs them. The parser was tested manually with an invalid circuit
  (`ghz_invalid_parse.txt`: two gates on one qubit in the same step), which it rejects with a
  clear error.

Tests pin down correctness before any optimisation:
- **Gates (`test_gates.py`):** each gate on a simple input gives what its matrix predicts.
- **Noise limits (`test_noise.py`):** with no noise X gives 1 and GHZ gives only 000 and 111;
  readout alone flips with probability p_readout; gate and idle noise flip with probability
  ⅔·p (within 5σ, fixed seed).
- **Invariants (`test_invariants.py`):** the state stays normalised; a gate followed by its
  inverse returns the original state.

## Part 2: validating the noise

### The experiment

Here we test to make sure the noise behaves as we expect.
We start with one qubit in 1 (X), leave it idle for m time steps, then measure; p_gate = 0, so the only errors are idle and readout (`idle_noise.py`). Run m = 0, 5, …, 60 with 10,000 shots each and record the fraction f of shots that read 1. Each m uses its own seed, so the points are independent.

### The expected curve
- Each shot reads 0 or 1, so the number of 1s is binomial.
- As the number of idle steps m grows, the random X/Y errors scramble the qubit, so it becomes equally likely to read 0 or 1: the fraction tends to 1/2. (More shots only make each point more precise; they do not move it towards 1/2.)
- Initially, at m = 0, we just have an X gate and a readout. The readout error is 2%, so the fraction starts at 0.98 and decays to 1/2.
- Per idle step, the bit flips with probability q = (2/3)·p_idle (X or Y flips it, Z does not; this comes from which idle error is chosen, not from gate noise).
- The function is f = 1/2 + d, where d is the distance above 1/2.
- One step maps d to (1 − 2q)·d, so after m steps
  d = (1/2)(1 − 2q)^m, starting from d = 1/2 (the qubit is 1).
- Readout flips the result with probability r = p_readout, which scales d by (1 − 2r).

  f(m) = 1/2 + 1/2 · (1 − 2r) · (1 − 2q)^m,   q = 2/3 · p_idle

The curve starts at 1 − r and decays to 1/2. It has the form C + A·B^m, with C = 1/2,
A = (1/2)(1 − 2r) and B = 1 − 2q. The two noise sources act differently:
- **Readout error (A)** scales the curve vertically: a larger r shrinks the whole curve towards
  1/2, but does not change how fast it decays.
- **Idle error (B)** sets the decay rate, a horizontal stretch: a smaller p_idle gives B closer
  to 1, so the curve approaches 1/2 more slowly.

This is why p_readout and p_idle can be separated in the fit: one sets the height, the other
the speed of the decay.

### Fit and uncertainty

- Error bars on each point: binomial, sqrt(f(1 − f)/N), with N = 10,000 shots. This is the standard deviation, sqrt(variance).
- p_readout is treated as known (it is an input of the simulation), so the fit has one
  free parameter, p_idle. It is fitted with `scipy.optimize.curve_fit`, weighted by the error bars (`absolute_sigma=True`).
- Result: injected p_idle = 0.05, fitted **0.04911 ± 0.00054**, 1.66σ from the injected
  value, so consistent.
- One could instead fit p_readout with p_idle fixed, or fit both.

![Idle decay](figures/idle_decay.png)

### Shot count

With N = 10,000 shots the error bar on each point is at most 0.005 (at f = 1/2), small
compared with the drop of the curve over the first 20–30 steps, and the fitted p_idle comes out to about 1% relative uncertainty. The error scales as 1/sqrt(N): 4× more shots halve it.

### Where the estimate stops being informative

- **Very small p_idle:** over m ≤ 60 the curve barely moves, so the decay is smaller than the error bars and p_idle is poorly determined. More steps or more shots are needed.
- **Very large p_idle:** the curve reaches 1/2 within a few steps, so only the first points carry information. At p_idle = 0.75, 1 − 2q = 0 and the qubit is fully randomised after one step; beyond that the model no longer describes a decay.
- **Small p_idle stretches the curve horizontally**, so it approaches 1/2 more slowly; this is
  why m must extend further when p_idle is small.
- **Large readout error:** a larger r makes the factor (1 − 2r) smaller, which shrinks the whole
  curve vertically towards 1/2; the decay becomes harder to see against the error bars, and at
  r = 1/2 there is nothing left to fit.

## Part 3: Making it fast

### 1. Hardware

Apple M3 Pro (MacBook Pro), 11 cores (5 performance + 6 efficiency), 16 MiB L2, 18 GB unified
memory with 150 GB/s published peak bandwidth; GPU not used. Details in `hardware.md`.

### 2. Reference version

`sim_reference.py` is an exact copy of `sim.py` at tag `v1-reference`: the baseline for every
speed-up (`benchmark.py --impl reference`) and the ground truth for the equivalence tests.

### 3. Benchmark circuit (brickwork)

`brickwork.py` generates the benchmark circuit from (n, D, seed). Each of the D layers
applies RY(θ) to every qubit with a random θ, then a TICK, then CNOTs on neighbouring pairs:
(0,1), (2,3), … in odd layers and (1,2), (3,4), … in even layers, then a TICK. The angles come from a seeded generator, so the same (n, D, seed) always gives the same circuit, and every version is benchmarked on identical input. At D = 20 a circuit has 20n RY gates and about 10(n − 1) CNOTs (590 gates at n = 20).



### 4. Method: profile, change one thing, measure

Each round: profile both regimes with cProfile, change the one function the profile points at, check the results against the reference (`test_equivalence.py`), benchmark both regimes, and record it (commit, log, profile, a row in `results/progress.md`), so each claim can be traced. Each cProfile output is saved and used to pick the next component to optimise. Close results were rechecked with 3 alternating runs of the old and new code, and the medians compared. If a change made performance worse, it was reverted, and the reason was recorded.

### 5. Optimisation rounds

Many-qubits time at n = 19 and speed-up against the reference (81 s); shots per second are in
the table below.

#### Initial - Reference
- The reference was extremely slow as the number of qubits or shots increased. The profile
  showed why: Python loops over all 2^n amplitudes in `apply_1q` (59%) and `apply_2q` (41%).
  This is the baseline (1×).

#### Optimization 1 & 2
- **Bottleneck:** the Python loops. Optimisation 1 vectorised `apply_1q` with NumPy on a
  reshaped view, but gave only 2.4× (33.6 s at n = 19): the `apply_2q` loop then took 97% of the time
  (Amdahl's law). Optimisation 2 vectorised `apply_2q` too, so no Python loop over amplitudes is
  left.
- This significantly helped: **1.84 s at n = 19 (44×)**.

#### Optimization 3a (Reverted)
- **Bottleneck:** `apply_1q` was 48% of the time, about 8 NumPy calls per gate.
- Attempted to apply it as one matrix multiplication. Because of the shape of the reshaped state
  (hundreds of thousands of tiny 2 × 2^k blocks), it was slower: n = 19 went from 1.59 s to
  2.25 s. Matrix multiplication is effective on a few large matrices, not on many small ones.
  Shots were slightly faster, but the many-qubits time was worse, so it was reverted.

#### Optimization 3b
- **Bottleneck:** CNOT went through the general 4×4 matrix (16 multiplies, 12 adds per group of
  four) although it only swaps two quarters of the state.
- Recognising what each gate does and replacing the matrix operation with a swap:
  **0.944 s at n = 19 (86×)**.

#### Optimization 3c
- **Bottleneck:** each `apply_1q` call created several temporary arrays (extra passes over memory).
- Replacing regular operators with in-place ones, for example `x = x + b` with `x += b`: 3–8%
  faster for n ≥ 19, but 21% slower for shots (small arrays), so it is used only for n ≥ 19.

#### Optimization 3c+
- **Bottleneck:** the noise errors X, Y, Z went through the general 2×2 matrix.
- Replacing the matrix with simple operations (swap, sign flip) for X, Y and Z: **0.925 s at n = 19 (88×)**, the same as 3c within noise, since these only run as rare errors.

#### Optimization 3d (Reverted)
- **Bottleneck:** RY is 400 of 590 gates and was done with complex arithmetic although its
  matrix is real.
- RY with real arithmetic cut its flops about 3× but was 16–22% slower (1.089 s at n = 19),
  because the kernel is memory-bound: the bytes moved did not change. Reverted.

#### Optimization 4: Numba
- **Bottleneck:** `apply_1q` was still 75–87% of the time: about 7 NumPy passes over the state
  per gate, and about 18 µs of call overhead per gate for shots.
- Numba compiles `apply_1q` into one loop: one pass, no temporaries, little overhead.
  **0.396 s at n = 19 (205×)**, and it reaches n = 29.

#### Correctness of optimised versions

Every round passed `test_equivalence.py` (72 tests) against the reference before it was benchmarked.

- **Noiseless: compared directly.** Final states (all 2^n amplitudes) must match the reference
  within 1e-12, over brickwork and random circuits using every gate, including gates on the
  lowest and highest qubits. The tests use small n, so the size-dependent paths tried in 3c and 3d
  (used only for n ≥ 19) were also checked at n = 19 and 20 (difference 0.0); both were later
  replaced by the numba kernel.
- **Noisy, same seed: identical counts.** The strictest check, but it relies on both versions
  drawing random numbers in the same order; an optimisation that reorders the draws (e.g.
  batching the noise) would fail it while still being correct.
- **Noisy, statistical: chi-squared.** 20,000 shots each with different seeds; the two count
  distributions must be consistent (p > 0.001). This does not depend on the order of the draws,
  so it stays valid after any such change.

#### Table and Plot

**Many shots:** n = 12, D = 20, p_gate = p_idle = 10⁻³, p_readout = 10⁻². Speed-up against the
reference (2.6 shots/s).

| Round | Change | Shots/s | Speed-up |
|---|---|---|---|
| Reference | Python loops over amplitudes | 2.6 | 1× |
| Optimisation 1 | Vectorise `apply_1q` | 6.1 | 2.3× |
| Optimisation 2 | Vectorise `apply_2q` | 109.0 | 42× |
| Optimisation 3a (reverted) | `apply_1q` as one matmul | 126.5 | 49× |
| Optimisation 3b | CNOT as a swap | 188.7 | 73× |
| Optimisation 3c | `apply_1q` in place (all sizes) | 152.3 | 59× |
| Optimisation 3c+ | In place for n ≥ 19 only; X, Y, Z as swaps/sign flips | 192.3 | 74× |
| Optimisation 3d (reverted) | RY and CZ kernels | 169.1 | 65× |
| Optimisation 4 | `apply_1q` as a numba loop | 506.2 | 195× |

Shots per second does not depend on the number of shots (each shot does the same work), so slower early versions were run with fewer shots and later ones with 10⁵. The reference at 10⁵ shots would take about
11 hours.

**Many qubits:** noiseless, 1 shot, D = 20, n from 4 until each version passes 300 s per run
(the final version reaches n = 29, the machine's memory limit; peak memory is in the Scaling
table). Reference points for n = 4–19 are the mean of two runs. Generated by `plot_scaling.py`.

![Many qubits: time against n](figures/qubits_scaling.png)

- **Parallel straight lines above n ≈ 14.** Every version grows about 2× per qubit (see
  Scaling); optimisations lower the line but cannot change the slope, because every gate still
  touches the whole state.
- **Flat line for Optimisation 4 below n ≈ 16** is about 0.14 s of loading numba's compiled code
  inside the timed call: a measurement artefact, not the algorithm.
- **Zig-zags below about 10 ms** are timer noise in single runs.

### 6. Scaling

#### Time growth vs n

Every gate touches the whole state of 2^n amplitudes, and the circuit has about 30n gates at
D = 20, so **time ≈ (gates) × 2^n**: each extra qubit costs about 2.1×. On the log-scale plot
this is a straight line, and each optimisation moves the line down without changing its slope.

Round 4 (numba), the final version, up to the machine limit:

| n | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 |
|---|---|---|---|---|---|---|---|---|---|---|
| Time (s) | 0.75 | 1.53 | 2.97 | 6.34 | 12.9 | 26.4 | 59.0 | 124 | 261 | 1080 |
| Ratio to n − 1 | 1.65 | 2.05 | 1.94 | 2.13 | 2.03 | 2.05 | 2.24 | 2.10 | 2.10 | 4.14 |
| Peak memory (GB) | 0.18 | 0.29 | 0.33 | 0.54 | 0.95 | 1.77 | 2.87 | 4.07 | 6.59 | 11.0 |
| Bandwidth (GB/s) | 24.4 | 24.9 | 27.0 | 26.5 | 27.2 | 27.6 | 25.7 | 25.5 | 25.1 | 12.6 |
| % of 150 GB/s peak | 16 | 17 | 18 | 18 | 18 | 18 | 17 | 17 | 17 | 8 |

- **n ≈ 17–20:** the state (16 × 2^n bytes) outgrows the 16 MiB L2 cache; the ratio rises to 2.
- **n = 21–28:** the state streams from main memory on every gate; the ratio holds at 2.0–2.2.
- **n = 29:** 4.1×: the process peaked at 11 GB of 18 GB (memory pressure). n = 30 needs 16 GiB
  (17.2 GB) for the state alone, so n = 29 is the limit of this machine.

#### Bytes, flops and bandwidth

- **Bytes per gate** (N = 2^n amplitudes, 16 bytes each, minimum read plus write): RY (numba
  loop) **32N**; CNOT (swap of two quarters, via a temporary copy) **24N**.
- **Flops per gate:** RY does 4 complex multiplies (6 flops) and 2 complex adds (2 flops) per
  pair, **14N**; CNOT only moves data, **0**.
- **Memory-bound:** RY does about **0.4 flops per byte**, far less than a core can compute per
  byte loaded, so the kernels wait on memory. At n = 24 the RY gates reach about 9 GFLOP/s on
  one core, well below its arithmetic capacity (peak flop rate not measured). Evidence: round 3d
  cut RY's flops about 3× with real arithmetic and got slower, because the bytes did not change.
- **Achieved bandwidth** (total bytes ÷ time, table above): **25–28 GB/s, about 17–18% of
  Apple's published 150 GB/s peak**, steady for n = 20–28; at n = 29 memory pressure halves it.

#### What stops it getting closer

- **One thread.** The numba loop uses one of 11 cores, which cannot saturate the memory system.
  This is the largest factor; multithreading is the fix.
- **Access pattern.** For high qubits the two halves of each pair are far apart in memory.
- **Extra traffic in CNOT.** The swap goes through a temporary copy of a quarter.
- **Theoretical peak.** 150 GB/s is Apple's figure, not a measurement; the achievable figure is
  lower, so the true fraction is higher.

These are estimates from byte counts and wall-clock times, not hardware counters.

#### Prediction at the largest n

- **Model:** every gate streams the whole state, so time(n + 1) = time(n) × 2 × gates(n + 1) /
  gates(n), with 20n RY gates plus the CNOTs (830 at n = 28, 860 at n = 29). Predicting each n
  from the previous one, it matches the measurements within 8% for n = 22–28, and within 2% for
  n = 27–28.
- **Prediction for n = 29:** 2 × 860 / 830 × 260.8 s ≈ **540 s**. **Measured: 1080 s**, twice
  the prediction. The model assumes bandwidth is the only limit; at n = 29 memory capacity took
  over (11 GB of 18 GB, compression and swapping), which the model does not include.
- **Note:** the n = 29 run was made before this prediction was written down. The prediction uses
  only n ≤ 28, but it is not a blind test; with more time I would predict first, then rerun.

#### Limiting resource in each regime

- **Many qubits: memory bandwidth (n = 20–28), then memory capacity (n = 29)**; see Scaling.
- **Many shots: Python overhead, not hardware.** At n = 12 the 64 KB state stays in cache, and a
  gate takes about 3.3 µs. The round 4 profile: numba `apply_1q` loop 41%, CNOT swap 17%,
  rebuilding the RY matrices every shot 14%, the rest Python bookkeeping (gate dispatch, noise
  draws, the shot loop). The hardware is far from its limits; reusing work across shots
  (caching RY matrices, batching shots) is the way forward.

### What I would do next with more time

In order of expected value, based on the profiles and the bandwidth numbers above:

1. **Multithreading (numba `prange`).** The many-qubits regime runs on one core at 17–18% of peak bandwidth. Splitting each gate's loop across cores is the most direct way to move closer to the memory limit.
2. **Reuse work across shots.** In the shots regime the time is Python overhead, not hardware. Building the RY matrices once per circuit instead of every shot (`ry()` is called 480,000 times in 2,000 shots) removes about 14%; moving CNOT into numba removes most of its 17%; running many shots together (one array per batch, noise drawn in bulk) would remove most of the per-gate Python cost.
3. **Migrate to C++.** A C++ architecture might be faster and more useful here than Python. So I would migrate to C++ and check the benchmarks there. 
4. **More benchmark iterations.** Most benchmarks here are single runs. Repeating each one
   several times would give a median and a spread (error bars) for every point: single runs
   are noisy, especially below about 10 ms, and two decisions here (3a, 3d) were only settled
   by 3-run rechecks after a single run looked ambiguous. I would also rerun all versions under
   identical settings, run longer to extend the large-n points, and write the large-n prediction
   down before running it.
5. **Wider testing.** Run the equivalence tests at large n as well, so size-dependent code paths are always covered, and run the Part 2 limits (very small and very large p_idle).
6. **Comparison with an established simulator** (e.g. Qiskit Aer) on the same circuits, and a GPU version using the M3's unified memory.
7. **Test on a different machine.** Rerun the benchmarks on other hardware to check the results.
8. **Size-dependent kernels.** Some changes helped one regime and hurt the other: in-place
   operations (3c) were 3–8% faster for large n but 21% slower for shots at n = 12. The fix was
   to pick the code path by state size (`INPLACE_MIN_SIZE = 2**19` in `apply_1q`, commit
   `b8508de`), later replaced by the numba kernel. I would look for more such crossovers, where
   the best method depends on n (cache size, call overhead), and choose the kernel accordingly.
