"""Tests for gate noise, idle noise and readout error.

Noisy output is random, so each test compares a measured fraction with
its expected value and allows for random variation: the result must be
within 5 standard deviations, sqrt(p * (1 - p) / shots). The seed is fixed,
so each test gives the same result every time it runs.

Run with::

    python3 -m pytest test_noise.py
"""

import math

from read_circuit import parse_circuit
from sim import sample_counts

SHOTS = 10_000
SEED = 1


def counts_for(text, p_gate=0.0, p_idle=0.0, p_readout=0.0, shots=SHOTS):
    """Parse circuit text and return the measured counts."""
    n_qubits, steps = parse_circuit(text)
    return sample_counts(n_qubits, steps, shots, p_gate, p_idle, p_readout, SEED)


def assert_close(fraction, expected, shots=SHOTS):
    """Check a measured fraction is within 5 standard deviations of expected."""
    sigma = math.sqrt(expected * (1 - expected) / shots)
    assert abs(fraction - expected) < 5 * sigma, (fraction, expected, sigma)


# ---- no noise ----

def test_no_noise_x_always_gives_1():
    counts = counts_for("QUBITS 1\nX 0\n", shots=1000)
    assert counts == {"1": 1000}


def test_no_noise_ghz_gives_only_000_and_111():
    with open("ghz.txt") as f:
        n_qubits, steps = parse_circuit(f.read())
    counts = sample_counts(n_qubits, steps, SHOTS, 0.0, 0.0, 0.0, SEED)
    assert set(counts) == {"000", "111"}
    assert_close(counts["000"] / SHOTS, 0.5)


# ---- readout error ----

def test_readout_only_x_gives_0_with_probability_p_readout():
    p = 0.1
    counts = counts_for("QUBITS 1\nX 0\n", p_readout=p)
    assert_close(counts.get("0", 0) / SHOTS, p)


# ---- gate noise ----

def test_gate_noise_flips_bit_with_probability_two_thirds_p():
    # After X the qubit is 1. An X or Y error flips it to 0; a Z error does not.
    p = 0.3
    counts = counts_for("QUBITS 1\nX 0\n", p_gate=p)
    assert_close(counts.get("0", 0) / SHOTS, 2 * p / 3)


# ---- idle noise ----

def test_idle_noise_flips_bit_with_probability_two_thirds_p():
    # Step 1: X on qubit 0. Step 2 is empty, so qubit 0 is idle once.
    p = 0.3
    counts = counts_for("QUBITS 1\nX 0\nTICK\nTICK\n", p_idle=p)
    assert_close(counts.get("0", 0) / SHOTS, 2 * p / 3)


def test_idle_noise_skips_qubits_used_by_a_gate():
    # The only step has X on qubit 0, so nothing is idle: even p_idle = 1
    # must cause no errors.
    counts = counts_for("QUBITS 1\nX 0\n", p_idle=1.0, shots=1000)
    assert counts == {"1": 1000}
