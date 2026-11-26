# Optimisation progress

One row per round: one change, tested against the reference, then benchmarked in both
regimes. Speed-ups are against the reference (`sim_reference.py`, tag `v1-reference`).

- Many qubits: noiseless, 1 shot, depth 20 (`benchmark.py qubits`); time at n = 19.
- Many shots: n = 12, depth 20, p_gate = p_idle = 1e-3, p_readout = 1e-2 (`benchmark.py shots`).

| Round | Commit | Change | Tests | n = 19 time | Largest n under 60 s | Shots/s | Speed-up vs reference | Log | Profile |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4189eb3 | Reference: Python loops over amplitudes | baseline (47 tests pass) | 81.0 s | 18 | 2.5 (500 shots) | 1× | reference_run.log | profile_0_reference_*.txt |
| 1 | b993111 | Vectorise `apply_1q` (reshaped view) | 47 pass; bit-for-bit identical to reference | 33.6 s | 19 | 6.1 (500 shots) | 2.4× | vec1q_run.log | profile_1_vec1q_*.txt |
| 2 | c422a37 | Vectorise `apply_2q` (reshaped view) | 47 pass; identical to reference (random 4×4: 1.8e-15 rounding) | 1.84 s | 23 | 109 (10,000 shots) | 44× | vec2q_run.log | profile_2_vec2q_*.txt |
| 3a | 2de3d6a | `apply_1q` as one matmul (`U @ view`) | 112 pass (72 equivalence) | 2.25 s (median of 3) | 23 | 126.5 (10,000 shots) | 36× | round3a_run.log | profile_3a_matmul1q_*.txt |
| 3b | c667dbe | CNOT as a swap of two quarters (`apply_cnot`); 3a reverted first | 112 pass (72 equivalence) | 0.944 s (median of 3) | 24 | 193.0 (10,000 shots) | 86× | round3b_run.log | profile_3b_cnot_*.txt |
| 3c | 72c9a88 | `apply_1q` in place (`*=`, `+=`; one copy, one temporary) | 112 pass (72 equivalence) | 0.891 s (median of 3) | 24 | 152.3 (10,000 shots) | 91× | round3c_run.log | profile_3c_inplace_*.txt |
| 3d | ee8102c–d74f4a9 | Special kernels: CZ, X, Y, Z (swap/sign) and RY (float view, real cos/sin) | 112 pass (72 equivalence) | 1.089 s (median of 3) | 24 (27 under 300 s) | 169.1 (100,000 shots) | 74× | round3d_run.log | — |
| 4 | 38cd325 | `apply_1q` as a compiled loop (numba `@njit`): one pass, no temporaries; CZ and RY kernels removed (7991f4f) | 112 pass (72 equivalence) | 0.396 s (median of 3) | 26 (29 under 300 s) | 506.2 (100,000 shots) | 205× | round4_run.log | profile_4_numba_*.txt |
