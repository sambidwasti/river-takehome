"""One simple test per gate: apply it to a basis state and check the result.

Run with::

    python3 -m pytest
"""

import numpy as np

from sim import GATES_1Q, GATES_2Q, ROTATIONS, apply_1q, apply_2q


def basis(n_qubits, i):
    """Return the state where only amplitude i is 1 (all qubits set as in i)."""
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[i] = 1
    return psi


# ---- fixed single-qubit gates (1 qubit: index 0 = |0>, index 1 = |1>) ----

def test_x_flips_0_to_1():
    psi = basis(1, 0)
    apply_1q(psi, GATES_1Q["X"], 0)
    assert np.allclose(psi, [0, 1])


def test_y_maps_0_to_i_times_1():
    psi = basis(1, 0)
    apply_1q(psi, GATES_1Q["Y"], 0)
    assert np.allclose(psi, [0, 1j])


def test_z_flips_sign_of_1():
    psi = basis(1, 1)
    apply_1q(psi, GATES_1Q["Z"], 0)
    assert np.allclose(psi, [0, -1])


def test_h_makes_equal_mix_from_0():
    psi = basis(1, 0)
    apply_1q(psi, GATES_1Q["H"], 0)
    assert np.allclose(psi, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_s_multiplies_1_by_i():
    psi = basis(1, 1)
    apply_1q(psi, GATES_1Q["S"], 0)
    assert np.allclose(psi, [0, 1j])


def test_t_multiplies_1_by_phase_pi_over_4():
    psi = basis(1, 1)
    apply_1q(psi, GATES_1Q["T"], 0)
    assert np.allclose(psi, [0, np.exp(1j * np.pi / 4)])


# ---- rotations ----

def test_rx_pi_maps_0_to_minus_i_times_1():
    psi = basis(1, 0)
    apply_1q(psi, ROTATIONS["RX"](np.pi), 0)
    assert np.allclose(psi, [0, -1j])


def test_ry_pi_maps_0_to_1():
    psi = basis(1, 0)
    apply_1q(psi, ROTATIONS["RY"](np.pi), 0)
    assert np.allclose(psi, [0, 1])


def test_rz_gives_0_the_phase_minus_theta_over_2():
    theta = np.pi / 2
    psi = basis(1, 0)
    apply_1q(psi, ROTATIONS["RZ"](theta), 0)
    assert np.allclose(psi, [np.exp(-1j * theta / 2), 0])


# ---- two-qubit gates (2 qubits: index = q0*1 + q1*2) ----

def test_cnot_flips_target_when_control_is_1():
    psi = basis(2, 1)  # q0 = 1, q1 = 0
    apply_2q(psi, GATES_2Q["CNOT"], 0, 1)  # control 0, target 1
    assert np.allclose(psi, basis(2, 3))  # q0 = 1, q1 = 1


def test_cz_flips_sign_when_both_are_1():
    psi = basis(2, 3)  # q0 = 1, q1 = 1
    apply_2q(psi, GATES_2Q["CZ"], 0, 1)
    assert np.allclose(psi, -basis(2, 3))
