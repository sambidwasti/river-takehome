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

## Notes

- Round 1 gave only 2.4×: `apply_1q` was 59% of the reference time, so the `apply_2q` loop
  (41%) still limited the total (Amdahl's law).
- Round 2: at n = 24 a run takes 78.7 s (about 0.11 s per gate on a 256 MB state), far below
  the 150 GB/s peak bandwidth; the many-shots regime is about 20 µs per gate at n = 12,
  mostly fixed per-call overhead.
