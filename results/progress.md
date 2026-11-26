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
| 3b | a8001df | CNOT as a swap of two quarters (`apply_cnot`); 3a reverted first | 112 pass (72 equivalence) | 0.944 s (median of 3) | 24 | 193.0 (10,000 shots) | 86× | round3b_run.log | profile_3b_cnot_*.txt |
| 3c | 652e82c | `apply_1q` in place (`*=`, `+=`; one copy, one temporary) | 112 pass (72 equivalence) | 0.891 s (median of 3) | 24 | 152.3 (10,000 shots) | 91× | round3c_run.log | profile_3c_inplace_*.txt |
| 3d | ee8102c–d74f4a9 | Special kernels: CZ, X, Y, Z (swap/sign) and RY (float view, real cos/sin) | 112 pass (72 equivalence) | 1.089 s (median of 3) | 24 (27 under 300 s) | 169.1 (100,000 shots) | 74× | round3d_run.log | — |
| 4 | 38cd325 | `apply_1q` as a compiled loop (numba `@njit`): one pass, no temporaries; CZ and RY kernels removed (7991f4f) | 112 pass (72 equivalence) | 0.396 s (median of 3) | 26 (29 under 300 s) | 506.2 (100,000 shots) | 205× | round4_run.log | profile_4_numba_*.txt |

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
- Round 3b: CNOT no longer goes through the general 4x4 `apply_2q`; it swaps the control = 1 quarters (one copy, two writes, no flops). Quick check against round 2, 3 runs each, alternating: n = 19 1.436 → 0.944 s (1.5×), n = 20 3.361 → 2.157 s (1.6×). Profile at n = 20: CNOT now 0.23 s (was 1.96 s in `apply_2q`); `apply_1q` is 84% of the time. Shots 109 → 193 shots/s (1.8×).
- Why `apply_1q` is slow (3b profile): it is 87% of the time at n = 20 and 75% in the shots run. Each gate
  does 9 separate NumPy operations (1 copy, 4 multiplies, 2 adds, 2 writes), each a full pass over half the
  state with a new temporary: about 160N bytes moved against the ideal 32N (N = 2^n amplitudes, 16 B each).
  For small k the slices are strided (k = 0: every other element), wasting half of each cache line. RY is
  real but stored as complex, so each multiply costs 6 flops instead of 2. In the shots run (n = 12, 2048
  amplitudes per half) the arithmetic is under 1 µs; the ~18 µs per call is fixed overhead of ~10 NumPy calls.
- Round 3c (in place): quick check against 3b, 3 runs each, alternating (medians, s):
  n = 17 0.234 → 0.247, 18 0.464 → 0.469, 19 0.966 → 0.891, 20 2.183 → 2.033, 21 3.968 → 4.077,
  22 9.373 → 9.029, 23 19.662 → 18.989, 24 42.436 → 39.292. So 3–8% faster from n = 22 up (and at
  n = 19–20), about the same or slightly slower at n = 17–18 and 21: removing temporaries cuts only part
  of the traffic, since about 7 passes over half the state remain. The shots regime got slower,
  confirmed by a recheck (n = 12, 2000 shots, 3 runs each, alternating): 3b 199.6, 194.4, 191.7 against
  in place 155.4, 153.9, 153.7 shots/s, medians 194.4 → 153.9 (21% slower). Likely cause: on small arrays
  the in-place writes into strided views cost more than fresh contiguous temporaries.
  Decision: keep in place only for large states. `apply_1q` now picks the path by size
  (`INPLACE_MIN_SIZE = 2**19`, i.e. n >= 19, where the gain starts); smaller states use the 3b code.
- Commit hashes for rounds 0–3c in the table are from before the 2026-11-26 history rewrite (PDFs removed
  from every commit). Current equivalents: 5f7bb7a → 4189eb3 (tag v1-reference), 93eba8b → b993111,
  bbb0548 → c422a37, 5d4799d → 2de3d6a, a8001df → c667dbe, 652e82c → 72c9a88. Log and profile headers
  keep the old hashes.
- Round 3d (special kernels: CZ ee8102c, X ac3b064, Y f4245f6, Z c1dfd2c, RY d74f4a9): first run with
  `--max-seconds 300`, reaching n = 27 in 429 s (peak 5.5 GB); 100,000 shots at 169.1 shots/s. Slower
  than 3c. Recheck (3 runs each, alternating, medians): n = 19 0.895 → 1.089 s, 20 2.010 → 2.386,
  22 8.906 → 10.450, 24 39.27 → 45.59 (16–22% slower); shots 192.1 → 168.7 shots/s (12% slower).
  The noiseless sweep only uses RY and CNOT, so the RY float-view kernel is the cause: the kernel is
  memory-bound, so 3× fewer flops gain nothing, while the float view and extra per-call work cost more.
  CZ and RY were removed (7991f4f). X, Y, Z alone versus 3c (medians): n = 19 0.904 → 0.925 s,
  20 2.047 → 2.051, 22 9.180 → 9.201, shots 191.3 → 192.5: no measurable difference, since X, Y, Z
  only run as rare noise errors.
- Round 4 (numba `apply_1q`): quick check against the NumPy version (b31565d), 3 runs each, alternating
  (medians): n = 19 0.900 → 0.396 s (2.3×), 20 2.034 → 0.751 (2.7×), 22 9.099 → 2.968 (3.1×); shots
  196.5 → 510.4 shots/s (2.6×). The gain grows with n: one pass over the state instead of about 7.
  Full sweep with `--max-seconds 300` reached n = 29 (the machine limit: 2^30 amplitudes would need 17 GiB
  of 18 GB). Up to n = 28 each qubit costs 2.0–2.2×; n = 29 jumped 4.1× (260.8 → 1079.8 s, peak 11.0 GB),
  most likely memory pressure (compression/swap), so beyond n = 28 memory size, not bandwidth, limits.
  Small n now has a floor of about 0.14 s and peak memory of about 123 MB: loading numba and the cached
  compiled code happens inside the timed call and the import adds memory. Profile at n = 20: the loop is
  0.43 s of 0.82 s; `apply_cnot` (0.20 s) is now about a quarter. Shots profile: the loop is 4.4 µs per
  call (was about 18 µs); rebuilding RY matrices (`ry` + `numpy.array`, about 1.0 s of 5.2 s) and
  `apply_cnot` (0.88 s) are now the next targets.
