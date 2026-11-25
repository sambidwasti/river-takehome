"""Equivalence tests: the optimised sim.py against sim_reference.py.

sim_reference.py is the Part 1 simulator (tag v1-reference), kept unchanged.
Every optimisation must give the same answers:

1. Noiseless, exact: the final states agree to within rounding (1e-12).
2. Noisy, same seed: the counts are identical, because both draw the same
   random numbers in the same order. (If an optimisation changes the order
   of the random draws, this test no longer applies; test 3 still does.)
3. Noisy, statistical: different seeds, many shots; a chi-squared test
   checks the two count distributions could be the same (p-value > 0.001).

Run with::

    python3 -m pytest test_equivalence.py
"""

import math

import numpy as np
import pytest
from scipy.stats import chi2_contingency

import sim
import sim_reference
from brickwork import brickwork_circuit
from read_circuit import parse_circuit

ATOL = 1e-12  # allowed rounding difference between the two states

FIXED_1Q = ["X", "Y", "Z", "H", "S", "T"]
ROTATIONS_1Q = ["RX", "RY", "RZ"]
GATES_2Q = ["CNOT", "CZ"]


def random_circuit(n_qubits, n_steps, seed):
    """Return circuit text using every gate type, on random qubits.

    Two-qubit gates use any ordered pair, so the control can be above or
    below the target, and the qubits need not be neighbours.
    """
    rng = np.random.default_rng(seed)
    lines = [f"QUBITS {n_qubits}"]
    for _ in range(n_steps):
        free = list(rng.permutation(n_qubits))
        while free:
            if len(free) >= 2 and rng.random() < 0.4:
                j, k = int(free.pop()), int(free.pop())
                lines.append(f"{rng.choice(GATES_2Q)} {j} {k}")
            else:
                q = int(free.pop())
                if rng.random() < 0.5:
                    lines.append(f"{rng.choice(FIXED_1Q)} {q}")
                else:
                    theta = rng.uniform(0, 2 * math.pi)
                    lines.append(f"{rng.choice(ROTATIONS_1Q)} {theta:.17g} {q}")
        lines.append("TICK")
    return "\n".join(lines) + "\n"


def both_states(text):
    """Run the circuit noiselessly through both simulators."""
    n_qubits, steps = parse_circuit(text)
    return sim.run(n_qubits, steps), sim_reference.run(n_qubits, steps)


# ---- 1. noiseless, exact ----

@pytest.mark.parametrize("n_qubits", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_brickwork_state_matches_reference(n_qubits, seed):
    new, ref = both_states(brickwork_circuit(n_qubits, 10, seed))
    np.testing.assert_allclose(new, ref, rtol=0, atol=ATOL)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_all_gates_state_matches_reference(n_qubits, seed):
    new, ref = both_states(random_circuit(n_qubits, 10, seed))
    np.testing.assert_allclose(new, ref, rtol=0, atol=ATOL)


@pytest.mark.parametrize("gate", GATES_2Q)
@pytest.mark.parametrize("j, k", [(0, 4), (4, 0), (1, 2), (2, 1), (0, 1), (3, 4)])
def test_2q_gate_on_edge_qubits(gate, j, k):
    """Both orders (control above and below), and the lowest and highest qubits."""
    # H on every qubit first, so every amplitude is non-zero
    text = "QUBITS 5\n" + "".join(f"H {q}\n" for q in range(5)) + "TICK\n"
    text += "T 0\nS 2\nRY 0.3 4\nTICK\n" + f"{gate} {j} {k}\n"
    new, ref = both_states(text)
    np.testing.assert_allclose(new, ref, rtol=0, atol=ATOL)


@pytest.mark.parametrize("gate", FIXED_1Q + ROTATIONS_1Q)
@pytest.mark.parametrize("q", [0, 2, 4])
def test_1q_gate_on_edge_qubits(gate, q):
    """Every one-qubit gate on the lowest, a middle and the highest qubit."""
    text = "QUBITS 5\n" + "".join(f"H {p}\n" for p in range(5)) + "TICK\n"
    text += f"{gate} 0.7 {q}\n" if gate in ROTATIONS_1Q else f"{gate} {q}\n"
    new, ref = both_states(text)
    np.testing.assert_allclose(new, ref, rtol=0, atol=ATOL)


# ---- 2. noisy, same seed: identical counts ----

NOISE = dict(p_gate=0.02, p_idle=0.01, p_readout=0.02)


@pytest.mark.parametrize("text", [
    brickwork_circuit(4, 6, 1),
    random_circuit(4, 6, 1),
], ids=["brickwork", "all_gates"])
def test_noisy_counts_identical_with_same_seed(text):
    n_qubits, steps = parse_circuit(text)
    args = (n_qubits, steps, 2000, NOISE["p_gate"], NOISE["p_idle"], NOISE["p_readout"], 7)
    assert sim.sample_counts(*args) == sim_reference.sample_counts(*args)


# ---- 3. noisy, statistical ----

def test_noisy_distributions_agree():
    """Different seeds: the two count distributions pass a chi-squared test."""
    n_qubits, steps = parse_circuit(random_circuit(3, 6, 2))
    shots = 20_000
    noise = (NOISE["p_gate"], NOISE["p_idle"], NOISE["p_readout"])
    new = sim.sample_counts(n_qubits, steps, shots, *noise, 11)
    ref = sim_reference.sample_counts(n_qubits, steps, shots, *noise, 12)

    keys = sorted(set(new) | set(ref))
    table = [[new.get(b, 0) for b in keys], [ref.get(b, 0) for b in keys]]
    p_value = chi2_contingency(table).pvalue
    assert p_value > 0.001, (p_value, new, ref)
