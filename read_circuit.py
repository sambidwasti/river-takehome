"""Read a circuit file into time steps.

This module is step 1 of the noisy simulator. It turns the challenge's
text format into a qubit count and a list of time steps.

File format
-----------

* The first line is ``QUBITS n``.
* Every other line is one gate, a ``TICK``, or a comment starting with ``#``.
* ``TICK`` ends a time step. Two TICKs in a row make an empty step.
* The end of the file closes the last step, even without a trailing TICK.

Example::

    QUBITS 3
    X 0
    TICK
    TICK
    H 2

gives ``n_qubits = 3`` and ``steps = [[("X", 0)], [], [("H", 2)]]``.

Gate tuples
-----------

* ``("X", 0)``: single-qubit gate, as ``(name, qubit)``.
* ``("RZ", 0.785, 0)``: rotation, as ``(name, angle, qubit)``. The angle is in radians.
* ``("CNOT", 0, 1)``: two-qubit gate, as ``(name, control, target)``.

Command line::

    python read_circuit.py circuits/parse_check.txt
"""

import sys

#: Gates whose first argument is an angle in radians.
ROTATIONS = {"RX", "RY", "RZ"}


def clean_lines(text):
    """Strip whitespace and remove blank lines and comments.

    :param text: Full contents of a circuit file.
    :type text: str
    :returns: The meaningful lines, in file order.
    :rtype: list[str]

    >>> clean_lines("QUBITS 1\\n# comment\\n\\n X 0 \\n")
    ['QUBITS 1', 'X 0']
    """
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def parse_header(line):
    """Read the qubit count from the ``QUBITS n`` line.

    :param line: The first meaningful line of the file.
    :type line: str
    :returns: The number of qubits.
    :rtype: int

    >>> parse_header("QUBITS 3")
    3
    """
    return int(line.split()[1])


def parse_gate(line):
    """Turn one gate line into a gate tuple. 

    :param line: A gate line such as ``"RZ 0.785 0"`` or ``"CNOT 0 1"``.
    :type line: str
    :returns: ``(name, angle, qubit)`` for rotations, otherwise
        ``(name, qubit, ...)`` with the control qubit first.
    :rtype: tuple

    3 Cases
    >>> parse_gate("X 0")
    ('X', 0)
    >>> parse_gate("RZ 0.785 0")
    ('RZ', 0.785, 0)
    >>> parse_gate("CNOT 0 1")
    ('CNOT', 0, 1)
    """
    parts = line.split()
    name = parts[0]
    if name in ROTATIONS:
        return (name, float(parts[1]), int(parts[2]))
    return (name, *(int(q) for q in parts[1:]))


def parse_circuit(text):
    """Parse circuit text into a qubit count and time steps.

    Each ``TICK`` closes the current step, even an empty one. The end of
    the text closes the last step, unless the text ended with a ``TICK``.
    Gates within one step must act on different qubits.

    :param text: Full contents of a circuit file.
    :type text: str
    :returns: ``(n_qubits, steps)``. Each step is a list of gate tuples
        (see :func:`parse_gate`).
    :rtype: tuple[int, list[list[tuple]]]
    :raises ValueError: If two gates in one step use the same qubit.

    >>> parse_circuit("QUBITS 3\\nX 0\\nTICK\\nTICK\\nH 2\\n")
    (3, [[('X', 0)], [], [('H', 2)]])
    >>> parse_circuit("QUBITS 2\\nX 0\\nH 0\\n")
    Traceback (most recent call last):
    ...
    ValueError: qubit 0 is used twice in step 1 (line 'H 0')
    """
    lines = clean_lines(text)
    n_qubits = parse_header(lines[0])

    steps = []
    current = []
    used = set()                    # qubits already used in the current step
    for line in lines[1:]:
        if line == "TICK":
            steps.append(current)   # close this step, even if it is empty
            current = []
            used = set()
        else:
            gate = parse_gate(line)
            qubits = (gate[2],) if gate[0] in ROTATIONS else gate[1:]
            for q in qubits:
                if q in used:
                    raise ValueError(
                        f"qubit {q} is used twice in step {len(steps) + 1} (line {line!r})"
                    )
                used.add(q)
            current.append(gate)

    # End of text closes the last step, unless a TICK already closed it.
    if lines[-1] != "TICK":
        steps.append(current)

    return n_qubits, steps


def read_circuit(path):
    """Read a circuit file from disk.

    :param path: Path to the circuit file.
    :type path: str
    :returns: ``(n_qubits, steps)``, as from :func:`parse_circuit`.
    :rtype: tuple[int, list[list[tuple]]]
    :raises FileNotFoundError: If ``path`` does not exist.
    """
    with open(path) as f:
        return parse_circuit(f.read())


def main(argv=None):
    """Command-line entry point: read a circuit file and print its steps.

    :param argv: Arguments without the program name. Defaults to
        ``sys.argv[1:]``. Expects one item: the circuit file path.
    :type argv: list[str] or None
    """
    argv = sys.argv[1:] if argv is None else argv
    n_qubits, steps = read_circuit(argv[0])
    print("n_qubits =", n_qubits)
    print("steps = [")
    for step in steps:
        print("   ", step, ",")
    print("]")


if __name__ == "__main__":
    main()