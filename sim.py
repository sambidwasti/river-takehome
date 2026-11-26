"""Noisy quantum circuit simulator: command-line entry point.

Usage::

    python sim.py circuit.txt --shots 1000 --p-gate 0.001 --p-idle 0.001 --p-readout 0.01 --seed 1
"""

import argparse

import numba
import numpy as np

from read_circuit import read_circuit

#: Fixed single-qubit gate matrices.
GATES_1Q = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
}


def rx(theta):
    """Return the RX(theta) matrix. theta is in radians."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    """Return the RY(theta) matrix. theta is in radians."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    """Return the RZ(theta) matrix. theta is in radians."""
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


#: Rotation gates: name -> function that builds the 2x2 matrix from the angle.
ROTATIONS = {"RX": rx, "RY": ry, "RZ": rz}

#: Two-qubit gate matrices. Row/column order is (first qubit, second qubit)
#: = 00, 01, 10, 11, with the first-listed qubit (the control) as the high bit.
GATES_2Q = {
    "CNOT": np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype=complex),
    "CZ": np.diag([1, 1, 1, -1]).astype(complex),
}


def zero_state(n_qubits):
    """Return the state with every qubit 0: ``psi[0] = 1``, all else 0.

    :param n_qubits: Number of qubits.
    :type n_qubits: int
    :rtype: numpy.ndarray

    >>> zero_state(1)
    array([1.+0.j, 0.+0.j])
    """
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1
    return psi


@numba.njit(cache=True)
def _apply_1q_loop(psi, U, k):
    """Compiled loop behind :func:`apply_1q` (numba, machine code).

    One pass over psi: each pair is read once, multiplied by U and written
    back. No copies, no temporary arrays.
    """
    stride = 1 << k                      # 2**k: distance between the pair
    u00, u01, u10, u11 = U[0, 0], U[0, 1], U[1, 0], U[1, 1]
    for block in range(0, psi.size, 2 * stride):   # blocks where bit k = 0, then 1
        for i in range(block, block + stride):     # every index with bit k = 0
            a0 = psi[i]
            a1 = psi[i + stride]
            psi[i] = u00 * a0 + u01 * a1
            psi[i + stride] = u10 * a0 + u11 * a1


def apply_1q(psi, U, k):
    """Apply a 2x2 gate U to qubit k, in place.

    For every index i whose bit k is 0, the pair ``(psi[i], psi[i + 2**k])``
    differs only in qubit k. Replace it with ``U @ (psi[i], psi[i + 2**k])``.

    Compiled loop (numba): one plain loop over the pairs, compiled to
    machine code. Each pair is read once and written once, so there is one
    pass over the state with no temporary arrays (the NumPy version made
    about 7 passes), and almost no per-call overhead. The first call
    compiles the loop (about a second); ``cache=True`` saves the compiled
    code so later runs skip that.

    :param psi: State vector; changed in place. Must be contiguous.
    :type psi: numpy.ndarray
    :param U: 2x2 gate matrix.
    :type U: numpy.ndarray
    :param k: Qubit the gate acts on.
    :type k: int

    >>> psi = zero_state(1)
    >>> apply_1q(psi, GATES_1Q["X"], 0)
    >>> psi
    array([0.+0.j, 1.+0.j])
    """
    _apply_1q_loop(psi, U, k)


def apply_x(psi, k):
    """Apply X to qubit k, in place.

    X swaps the amplitudes with qubit k = 0 and qubit k = 1. So this swaps
    the two halves of the reshaped view: no matrix and no arithmetic.

    Uses the same reshaped view as :func:`apply_1q`.

    :param psi: State vector; changed in place. Must be contiguous.
    :type psi: numpy.ndarray
    :param k: Qubit the gate acts on.
    :type k: int

    >>> psi = zero_state(1)
    >>> apply_x(psi, 0)
    >>> psi
    array([0.+0.j, 1.+0.j])
    """
    view = psi.reshape(-1, 2, 2**k)  # a view of psi
    a, b = view[:, 0, :], view[:, 1, :]
    tmp = a.copy()   # copy: a is overwritten next
    a[...] = b
    b[...] = tmp


def apply_y(psi, k):
    """Apply Y to qubit k, in place.

    Y = [[0, -i], [i, 0]] is a swap with a factor: the new qubit k = 0
    half is -i times the old qubit k = 1 half, and the new qubit k = 1
    half is i times the old qubit k = 0 half. Two multiplies and no adds,
    instead of the general four multiplies and two adds.

    Uses the same reshaped view as :func:`apply_1q`.

    :param psi: State vector; changed in place. Must be contiguous.
    :type psi: numpy.ndarray
    :param k: Qubit the gate acts on.
    :type k: int

    >>> psi = zero_state(1)
    >>> apply_y(psi, 0)
    >>> psi
    array([0.+0.j, 0.+1.j])
    """
    view = psi.reshape(-1, 2, 2**k)  # a view of psi
    a, b = view[:, 0, :], view[:, 1, :]
    tmp = a.copy()      # old qubit-0 half
    a[...] = -1j * b    # new 0-half = -i x old 1-half
    tmp *= 1j           # in place
    b[...] = tmp        # new 1-half = i x old 0-half


def apply_z(psi, k):
    """Apply Z to qubit k, in place.

    Z = [[1, 0], [0, -1]] leaves the qubit k = 0 half alone and flips the
    sign of the qubit k = 1 half: one in-place pass over half the state,
    no copy and no adds.

    Uses the same reshaped view as :func:`apply_1q`.

    :param psi: State vector; changed in place. Must be contiguous.
    :type psi: numpy.ndarray
    :param k: Qubit the gate acts on.
    :type k: int

    >>> psi = zero_state(1)
    >>> psi[0], psi[1] = 0, 1      # qubit 0 is 1
    >>> apply_z(psi, 0)
    >>> psi
    array([ 0.+0.j, -1.+0.j])
    """
    view = psi.reshape(-1, 2, 2**k)  # a view of psi
    view[:, 1, :] *= -1              # qubit k = 1: flip the sign


def apply_2q(psi, M, j, k):
    """Apply a 4x4 gate M to qubits j and k, in place.

    For every index i whose bits j and k are both 0, take the group
    ``(psi[i], psi[i + 2**k], psi[i + 2**j], psi[i + 2**j + 2**k])``
    and replace it with ``M @ group``. j is the first-listed qubit (the
    control for CNOT).

    Vectorised: with hi = max(j, k) and lo = min(j, k), reshape psi (a
    view, no copy) to (2**(n-hi-1), 2, 2**(hi-lo-1), 2, 2**lo). Axis 1 is
    bit hi and axis 3 is bit lo, so fixing those two axes picks out one
    of the four members of every group at once. Each new member is then
    a weighted sum of the four old ones, done with NumPy operations
    instead of a Python loop over 2**n indices.

    :param psi: State vector; changed in place. Must be contiguous.
    :type psi: numpy.ndarray
    :param M: 4x4 gate matrix.
    :type M: numpy.ndarray
    :param j: First-listed qubit.
    :type j: int
    :param k: Second-listed qubit.
    :type k: int
    """
    hi, lo = max(j, k), min(j, k)
    view = psi.reshape(-1, 2, 2**(hi - lo - 1), 2, 2**lo)  # a view of psi

    def part(bit_j, bit_k):
        """All amplitudes with qubit j = bit_j and qubit k = bit_k (a view)."""
        bit_hi, bit_lo = (bit_j, bit_k) if j > k else (bit_k, bit_j)
        return view[:, bit_hi, :, bit_lo, :]

    # Same order as the matrix rows: (j, k) = (0,0), (0,1), (1,0), (1,1).
    bits = ((0, 0), (0, 1), (1, 0), (1, 1))
    old = [part(bj, bk).copy() for bj, bk in bits]  # copies: psi is overwritten below
    for r, (bj, bk) in enumerate(bits):
        part(bj, bk)[...] = (M[r, 0] * old[0] + M[r, 1] * old[1]
                             + M[r, 2] * old[2] + M[r, 3] * old[3])


def apply_cnot(psi, c, t):
    """Apply CNOT with control c and target t, in place.

    Where qubit c is 0 nothing changes. Where qubit c is 1, the amplitudes
    with t = 0 and t = 1 swap places. So this is a swap of two quarters of
    the state: no multiplications, and the other half is never touched.

    Uses the same reshaped view as :func:`apply_2q`.

    :param psi: State vector; changed in place. Must be contiguous.
    :type psi: numpy.ndarray
    :param c: Control qubit.
    :type c: int
    :param t: Target qubit.
    :type t: int

    >>> psi = zero_state(2)
    >>> psi[0], psi[2] = 0, 1      # qubit 1 (the control) is 1
    >>> apply_cnot(psi, 1, 0)
    >>> psi
    array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])
    """
    hi, lo = max(c, t), min(c, t)
    view = psi.reshape(-1, 2, 2**(hi - lo - 1), 2, 2**lo)  # a view of psi
    if c > t:
        a, b = view[:, 1, :, 0, :], view[:, 1, :, 1, :]  # control 1; target 0, 1
    else:
        a, b = view[:, 0, :, 1, :], view[:, 1, :, 1, :]
    tmp = a.copy()   # copy: a is overwritten next
    a[...] = b
    b[...] = tmp


def apply_gate(psi, gate):
    """Apply one gate tuple to the state, in place.

    :param psi: State vector; changed in place.
    :type psi: numpy.ndarray
    :param gate: A gate tuple, as from :func:`read_circuit.parse_gate`.
    :type gate: tuple
    :raises ValueError: For an unknown gate name.
    """
    name = gate[0]
    if name == "X":
        apply_x(psi, gate[1])              # a swap: faster than the 2x2 matrix
    elif name == "Y":
        apply_y(psi, gate[1])              # a swap with factors -i, i
    elif name == "Z":
        apply_z(psi, gate[1])              # a sign flip on half the state
    elif name in GATES_1Q:
        apply_1q(psi, GATES_1Q[name], gate[1])
    elif name in ROTATIONS:
        apply_1q(psi, ROTATIONS[name](gate[1]), gate[2])
    elif name == "CNOT":
        apply_cnot(psi, gate[1], gate[2])  # a swap: faster than the 4x4 matrix
    elif name in GATES_2Q:
        apply_2q(psi, GATES_2Q[name], gate[1], gate[2])
    else:
        raise ValueError(f"unknown gate {name}")


def qubits_of(gate):
    """Return the qubits a gate acts on.

    >>> qubits_of(("X", 0))
    (0,)
    >>> qubits_of(("RZ", 0.785, 2))
    (2,)
    >>> qubits_of(("CNOT", 0, 1))
    (0, 1)
    """
    if gate[0] in ROTATIONS:
        return (gate[2],)
    return tuple(gate[1:])


def run(n_qubits, steps):
    """Run the circuit with no noise and return the final state.

    :param n_qubits: Number of qubits.
    :type n_qubits: int
    :param steps: Time steps, as from :func:`read_circuit.parse_circuit`.
    :type steps: list[list[tuple]]
    :rtype: numpy.ndarray
    """
    psi = zero_state(n_qubits)
    for step in steps:
        for gate in step:
            apply_gate(psi, gate)
    return psi


#: The three possible errors, picked uniformly at random.
PAULIS = ("X", "Y", "Z")


def maybe_error(psi, q, p, rng):
    """With probability p, apply a random X, Y or Z to qubit q, in place.

    :param psi: State vector; changed in place.
    :type psi: numpy.ndarray
    :param q: Qubit that may get the error.
    :type q: int
    :param p: Error probability.
    :type p: float
    :param rng: Random number generator.
    :type rng: numpy.random.Generator
    """
    if rng.random() < p:
        pauli = PAULIS[rng.integers(3)]
        if pauli == "X":
            apply_x(psi, q)                # a swap: faster than the 2x2 matrix
        elif pauli == "Y":
            apply_y(psi, q)                # a swap with factors -i, i
        else:
            apply_z(psi, q)                # Z: a sign flip on half the state


def one_shot(n_qubits, steps, p_gate, p_idle, p_readout, rng):
    """Run the circuit once with fresh random errors and measure it.

    * Gate noise: after each gate, each qubit it acted on may get an error
      (probability ``p_gate``).
    * Idle noise: in each step, each qubit no gate touched may get an error
      (probability ``p_idle``).
    * Measurement: pick one index i with probability ``|psi[i]|**2``.
    * Readout error: flip each measured bit with probability ``p_readout``.

    :param n_qubits: Number of qubits.
    :type n_qubits: int
    :param steps: Time steps, as from :func:`read_circuit.parse_circuit`.
    :type steps: list[list[tuple]]
    :param p_gate: Gate error probability.
    :type p_gate: float
    :param p_idle: Idle error probability.
    :type p_idle: float
    :param p_readout: Readout error probability.
    :type p_readout: float
    :param rng: Random number generator.
    :type rng: numpy.random.Generator
    :returns: The measured outcome, as an index (bit q = qubit q).
    :rtype: int
    """
    psi = zero_state(n_qubits)
    for step in steps:
        touched = set()
        for gate in step:
            apply_gate(psi, gate)
            for q in qubits_of(gate):
                maybe_error(psi, q, p_gate, rng)
                touched.add(q)
        for q in range(n_qubits):
            if q not in touched:  # idle in this step
                maybe_error(psi, q, p_idle, rng)

    probs = np.abs(psi) ** 2
    outcome = int(rng.choice(2**n_qubits, p=probs / probs.sum()))
    for q in range(n_qubits):
        if rng.random() < p_readout:
            outcome ^= 1 << q  # flip bit q
    return outcome


def sample_counts(n_qubits, steps, shots, p_gate, p_idle, p_readout, seed):
    """Run many shots and count the measured bitstrings.

    Bitstrings are written most significant bit first, so qubit 0 is the
    rightmost character.

    :param shots: Number of shots.
    :type shots: int
    :param seed: Random seed, or None for an unseeded run.
    :type seed: int or None
    :returns: Bitstring -> count, in bitstring order.
    :rtype: dict[str, int]

    The other parameters are as for :func:`one_shot`.
    """
    rng = np.random.default_rng(seed)
    counts = {}
    for _ in range(shots):
        outcome = one_shot(n_qubits, steps, p_gate, p_idle, p_readout, rng)
        key = format(outcome, f"0{n_qubits}b")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def parse_args(argv=None):
    """Read the command-line parameters.

    :param argv: Arguments without the program name. Defaults to ``sys.argv[1:]``.
    :type argv: list[str] or None
    :returns: The parsed parameters: ``circuit``, ``shots``, ``p_gate``,
        ``p_idle``, ``p_readout`` and ``seed``.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Noisy quantum circuit simulator")
    parser.add_argument("circuit", help="path to the circuit file")
    parser.add_argument("--shots", type=int, default=1000, help="number of shots (default: 1000)")
    parser.add_argument("--p-gate", type=float, default=0.0, help="gate error probability (default: 0)")
    parser.add_argument("--p-idle", type=float, default=0.0, help="idle error probability (default: 0)")
    parser.add_argument("--p-readout", type=float, default=0.0, help="readout error probability (default: 0)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: none)")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the simulator from the command line.

    :param argv: Arguments without the program name. Defaults to ``sys.argv[1:]``.
    :type argv: list[str] or None
    """
    args = parse_args(argv)
    print("circuit   =", args.circuit)
    print("shots     =", args.shots)
    print("p_gate    =", args.p_gate)
    print("p_idle    =", args.p_idle)
    print("p_readout =", args.p_readout)
    print("seed      =", args.seed)

    n_qubits, steps = read_circuit(args.circuit)
    print("n_qubits  =", n_qubits)
    print("steps     =", steps)
    
    # psi = run(n_qubits, steps)
    # print("state     =", psi)

    counts = sample_counts(n_qubits, steps, args.shots, args.p_gate,
                           args.p_idle, args.p_readout, args.seed)
    print("counts    =", counts)


if __name__ == "__main__":
    main()