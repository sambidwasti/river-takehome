"""Part 3 benchmark circuit: a brickwork circuit on n qubits with depth D.

Each layer (numbered 1 to D):
  * RY(theta) on every qubit, with a random theta, then a TICK
  * CNOTs on neighbouring pairs, then a TICK:
    odd layers  (1, 3, ...): (0,1), (2,3), ...
    even layers (2, 4, ...): (1,2), (3,4), ...

The angles come from a seeded random number generator, so the same
(n, depth, seed) always gives the same circuit.

Write a circuit file with::

    python3 brickwork.py 12 20 --seed 1 > brickwork_n12_d20.txt
"""

import argparse
import math

import numpy as np


def brickwork_circuit(n_qubits, depth, seed):
    """Return the brickwork circuit as text, in the simulator's circuit format."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * math.pi, size=(depth, n_qubits))

    lines = [f"QUBITS {n_qubits}"]
    for layer in range(1, depth + 1):
        for q in range(n_qubits):
            lines.append(f"RY {angles[layer - 1, q]:.17g} {q}")
        lines.append("TICK")
        first = 0 if layer % 2 == 1 else 1   # odd layers start at qubit 0, even at 1
        for q in range(first, n_qubits - 1, 2):
            lines.append(f"CNOT {q} {q + 1}")
        lines.append("TICK")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Write a brickwork benchmark circuit.")
    parser.add_argument("n_qubits", type=int)
    parser.add_argument("depth", type=int)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    print(brickwork_circuit(args.n_qubits, args.depth, args.seed), end="")


if __name__ == "__main__":
    main()
