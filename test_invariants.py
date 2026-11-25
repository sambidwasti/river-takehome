"""Invariant tests: properties that must hold for any state and any gate.

* Normalisation: the squared amplitudes always sum to 1.
* Gate then inverse: applying a gate and then its inverse gives back the
  starting state.

Each test starts from a random 3-qubit state, so it checks every amplitude,
not only a simple basis state.

Run with::

    python3 -m pytest test_invariants.py
"""

import numpy as np
import pytest

from read_circuit import read_circuit
from sim import (GATES_1Q, GATES_2Q, ROTATIONS, apply_1q, apply_2q,
                 apply_gate, maybe_error, qubits_of, zero_state)

N_QUBITS = 3
THETA = 0.7  # any angle works; this one is not a special value


def random_state(seed=0):
    """Return a random normalised state on N_QUBITS qubits."""
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**N_QUBITS) + 1j * rng.normal(size=2**N_QUBITS)
    return psi / np.linalg.norm(psi)


def norm(psi):
    """Return the sum of the squared amplitudes."""
    return np.sum(np.abs(psi) ** 2)


# Every gate, as a gate tuple on 3 qubits. Single-qubit gates act on
# qubit 1 (the middle bit). Two-qubit gates use qubits 2 and 0, so the
# control is the higher bit and the qubits are not neighbours.
ALL_GATES = [
    ("X", 1), ("Y", 1), ("Z", 1), ("H", 1), ("S", 1), ("T", 1),
    ("RX", THETA, 1), ("RY", THETA, 1), ("RZ", THETA, 1),
    ("CNOT", 2, 0), ("CZ", 2, 0),
]


# ---- normalisation ----

@pytest.mark.parametrize("gate", ALL_GATES, ids=lambda g: g[0])
def test_gate_keeps_state_normalised(gate):
    psi = random_state()
    apply_gate(psi, gate)
    assert np.isclose(norm(psi), 1)


def test_noisy_circuit_keeps_state_normalised():
    # Run parse_check.txt with an error on every qubit after every gate and
    # in every idle step (p = 1), then check the norm.
    n_qubits, steps = read_circuit("parse_check.txt")
    rng = np.random.default_rng(1)
    psi = zero_state(n_qubits)
    for step in steps:
        touched = set()
        for gate in step:
            apply_gate(psi, gate)
            for q in qubits_of(gate):
                maybe_error(psi, q, 1.0, rng)
                touched.add(q)
        for q in range(n_qubits):
            if q not in touched:
                maybe_error(psi, q, 1.0, rng)
    assert np.isclose(norm(psi), 1)


# ---- gate then inverse ----

def inverse_1q(name):
    """Return the inverse of a fixed single-qubit gate.

    X, Y, Z and H are their own inverse. S and T are not: their inverse is
    the conjugate transpose (S-dagger, T-dagger).
    """
    U = GATES_1Q[name]
    return U.conj().T


@pytest.mark.parametrize("name", ["X", "Y", "Z", "H", "S", "T"])
def test_fixed_gate_then_inverse_gives_back_state(name):
    psi = random_state()
    start = psi.copy()
    apply_1q(psi, GATES_1Q[name], 1)
    apply_1q(psi, inverse_1q(name), 1)
    assert np.allclose(psi, start)


@pytest.mark.parametrize("name", ["RX", "RY", "RZ"])
def test_rotation_then_minus_angle_gives_back_state(name):
    psi = random_state()
    start = psi.copy()
    apply_1q(psi, ROTATIONS[name](THETA), 1)
    apply_1q(psi, ROTATIONS[name](-THETA), 1)  # inverse of R(theta) is R(-theta)
    assert np.allclose(psi, start)


@pytest.mark.parametrize("name", ["CNOT", "CZ"])
def test_two_qubit_gate_twice_gives_back_state(name):
    # CNOT and CZ are their own inverse.
    psi = random_state()
    start = psi.copy()
    apply_2q(psi, GATES_2Q[name], 2, 0)
    apply_2q(psi, GATES_2Q[name], 2, 0)
    assert np.allclose(psi, start)
