"""Read and write the fitted coefficients of a ChIMES params.txt (2-body + 3-body, one atom type).

Coefficient order is the design-matrix column order used throughout (chimes_lsq): the O2 pair coefficients, then the
unique 3-body coefficients by "param index". Writing copies a template params.txt and replaces only the coefficient
values, so every hyperparameter (orders, cutoffs, Morse lambda, smooth cutoff) is inherited from the template.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_PAIR_HDR = re.compile(r"^PAIRTYPE PARAMS:")
_TRIP_HDR = re.compile(r"^\s*TRIPLETTYPE PARAMS:")
_PAIR_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s*$")
_TRIP_ROW = re.compile(r"^(\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+)(\d+)(\s+)(\S+)\s*$")


def _scan(lines):
    """yield (kind, line_no, index, match) for every coefficient line."""
    section = None
    for i, line in enumerate(lines):
        if _PAIR_HDR.match(line):
            section = "pair"; continue
        if _TRIP_HDR.match(line):
            section = "trip"; continue
        if line.startswith("PAIRMAPS") or line.startswith("TRIPMAPS"):
            section = None
        if section == "pair":
            m = _PAIR_ROW.match(line)
            if m:
                yield "pair", i, int(m.group(1)), m
            elif line.strip().startswith("TRIPLET") or line.strip() == "":
                if line.strip().startswith("TRIPLET"):
                    section = None
        elif section == "trip":
            m = _TRIP_ROW.match(line)
            if m:
                yield "trip", i, int(m.group(2)), m


def read_params(path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    pair, trip = {}, {}
    for kind, _, idx, m in _scan(lines):
        if kind == "pair":
            pair[idx] = float(m.group(2))
        else:
            v = float(m.group(4))
            assert trip.setdefault(idx, v) == v, f"inconsistent equivalent 3-body values at param index {idx}"
    assert sorted(pair) == list(range(len(pair))) and sorted(trip) == list(range(len(trip)))
    return np.array([pair[i] for i in range(len(pair))] + [trip[i] for i in range(len(trip))])


def write_params(template, coefs, out, comment: str = "") -> None:
    lines = Path(template).read_text().splitlines()
    n2 = sum(1 for k, *_ in _scan(lines) if k == "pair")
    coefs = np.asarray(coefs, float)
    n3 = len({idx for k, _, idx, _ in _scan(lines) if k == "trip"})
    assert coefs.size == n2 + n3, f"{coefs.size} coefficients for a {n2}+{n3} template"
    for kind, i, idx, m in list(_scan(lines)):
        if kind == "pair":
            lines[i] = f"{idx:3d}  {coefs[idx]: .13e}"
        else:
            lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{coefs[n2 + idx]: .13e}"
    if comment:
        lines.insert(0, f"! {comment}")
    Path(out).write_text("\n".join(lines) + "\n")
