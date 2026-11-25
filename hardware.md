# Hardware

All benchmarks were run on this machine.

| Item | Value |
|---|---|
| Machine | MacBook Pro (Mac15,6) |
| CPU | Apple M3 Pro |
| Cores | 11 (5 performance + 6 efficiency), no SMT |
| L1 data cache | 128 KiB per performance core, 64 KiB per efficiency core |
| L2 cache | 16 MiB shared by the performance cluster, 4 MiB shared by the efficiency cluster |
| Cache line | 128 bytes |
| Memory | 18 GB unified (LPDDR5) |
| Peak memory bandwidth | 150 GB/s (Apple's published figure for the M3 Pro; theoretical, not measured) |
| GPU | Apple M3 Pro, 14 cores (not used) |
| OS | macOS 27.2.1 |
| Python | 3.12.2 |
| NumPy | 1.26.4 (BLAS: OpenBLAS) |

## What this means for the benchmarks

- **State vector size:** 16 bytes × 2^n. 2^20 amplitudes (n = 20) is 16 MiB, the size of the
  performance-cluster L2. From about n = 20, the state no longer fits in cache, so every gate
  has to stream it from main memory.
- **Largest n:** n = 29 needs 8 GiB and n = 30 needs 16 GiB. With 18 GB in total, and room
  needed for the OS and any temporary copies, n = 29 is the practical limit.
- **Peak bandwidth** is the reference for the performance model (checklist item 10).
