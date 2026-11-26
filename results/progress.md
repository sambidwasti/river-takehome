# Optimisation progress

One row per round: one change, tested against the reference, then benchmarked in both
regimes. Speed-ups are against the reference (`sim_reference.py`, tag `v1-reference`).

- Many qubits: noiseless, 1 shot, depth 20 (`benchmark.py qubits`); time at n = 19.
- Many shots: n = 12, depth 20, p_gate = p_idle = 1e-3, p_readout = 1e-2 (`benchmark.py shots`).

| Round | Commit | Change | Tests | n = 19 time | Largest n under 60 s | Shots/s | Speed-up vs reference | Log | Profile |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5f7bb7a | Reference: Python loops over amplitudes | baseline (47 tests pass) | 81.0 s | 18 | 2.5 (500 shots) | 1× | reference_run.log | profile_0_reference_*.txt |
| 1 | 93eba8b | Vectorise `apply_1q` (reshaped view) | 47 pass; bit-for-bit identical to reference | 33.6 s | 19 | 6.1 (500 shots) | 2.4× | vec1q_run.log | profile_1_vec1q_*.txt |
| 2 | bbb0548 | Vectorise `apply_2q` (reshaped view) | 47 pass; identical to reference (random 4×4: 1.8e-15 rounding) | 1.84 s | 23 | 109 (10,000 shots) | 44× | vec2q_run.log | profile_2_vec2q_*.txt |
| 3a | 5d4799d | `apply_1q` as one matmul (`U @ view`) | 112 pass (72 equivalence) | 2.25 s (median of 3) | 23 | 126.5 (10,000 shots) | 36× | round3a_run.log | profile_3a_matmul1q_*.txt |

## Notes

- Round 1 gave only 2.4×: `apply_1q` was 59% of the reference time, so the `apply_2q` loop
  (41%) still limited the total (Amdahl's law).
- Round 2: at n = 24 a run takes 78.7 s (about 0.11 s per gate on a 256 MB state), far below
  the 150 GB/s peak bandwidth; the many-shots regime is about 20 µs per gate at n = 12,
  mostly fixed per-call overhead.
- Round 3a was slower than round 2 in the many-qubits regime. A recheck (3 runs each, alternating) at
  n = 17–20 showed 3a losing every pair; at n = 19 the median was 2.25 s against 1.59 s. Possible reason:
  `U @ view` multiplies a stack of many small (2 × 2^k) matrices one at a time; for small k (e.g. qubit 0,
  2 × 1 blocks) that is hundreds of thousands of tiny multiplications, slower than the elementwise version.
  The shots regime improved about 16% (109 → 126.5 shots/s). Kept as a record; to be reverted or replaced.
