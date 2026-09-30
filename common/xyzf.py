"""Minimal readers for ChIMES .xyzf trajectories (plain or gzip) and VASP OUTCAR forces."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def read_lines(path: Path) -> list[str]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return f.read().splitlines()


def frame_offsets(lines: list[str]) -> np.ndarray:
    """Line index of the atom-count line of every frame, plus the end."""
    off, i = [0], 0
    while i < len(lines):
        i += int(lines[i]) + 2
        off.append(i)
    return np.array(off)


def frame(lines: list[str], start: int):
    """(cell 3x3, positions n x 3) of the frame whose atom-count line is lines[start]."""
    n = int(lines[start]); c = lines[start + 1].split()
    cell = np.array(c[1:10], float).reshape(3, 3)
    x = np.array([l.split()[1:4] for l in lines[start + 2:start + 2 + n]], float)
    return cell, x


def outcar_forces(path: Path) -> np.ndarray:
    """Forces (eV/A) of the last TOTAL-FORCE block of a VASP OUTCAR."""
    L = Path(path).read_text().splitlines()
    i = max(k for k, l in enumerate(L) if "TOTAL-FORCE" in l) + 2
    F = []
    while not L[i].startswith(" ---"):
        F.append(L[i].split()[3:6]); i += 1
    return np.array(F, float)


def outcar_done(path: Path) -> bool:
    p = Path(path)
    return p.exists() and "General timing" in p.read_text()


def load_dft_forces(path: Path) -> dict:
    """data/dft_forces.npz -> {frame: (n_atoms, 3) forces in eV/A} (written by md_audit/05_collect_dft.py)."""
    z = np.load(path)
    off = np.r_[0, np.cumsum(z["n_atoms"])]
    return {int(f): z["forces"][off[i]:off[i + 1]] for i, f in enumerate(z["frame"])}
