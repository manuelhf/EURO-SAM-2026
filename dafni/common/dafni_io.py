"""Small helpers shared by the three DAFNI entry points.

DAFNI conventions (https://docs.secure.dafni.rl.ac.uk/docs/Reference/model-definition-reference):
  * parameters arrive as environment variables, always as strings
    (booleans as "True"/"False");
  * dataslots are mounted read-only under /data/inputs/<slot>, and outputs of
    an earlier workflow step under /data/inputs/<step-name>;
  * anything written to /data/outputs is kept as the run's output dataset.

INPUTS_DIR / OUTPUTS_DIR override those paths for local runs.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

INPUTS = Path(os.environ.get("INPUTS_DIR", "/data/inputs"))
OUTPUTS = Path(os.environ.get("OUTPUTS_DIR", "/data/outputs"))


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
def _raw(name: str) -> str | None:
    v = os.environ.get(name)
    return None if v is None or v.strip() == "" else v.strip()


def env_str(name: str, default: str, options: tuple[str, ...] | None = None) -> str:
    v = _raw(name) or default
    if options and v not in options:
        raise ValueError(f"{name} must be one of {options}, got {v!r}")
    return v


def env_int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    v = _raw(name)
    x = default if v is None else int(float(v))
    _check_range(name, x, lo, hi)
    return x


def env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    v = _raw(name)
    x = default if v is None else float(v)
    _check_range(name, x, lo, hi)
    return x


def env_bool(name: str, default: bool) -> bool:
    v = _raw(name)
    if v is None:
        return default
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"{name} must be true/false, got {v!r}")


def env_floats(name: str, default: tuple[float, ...], n: int | None = None) -> tuple[float, ...]:
    """Comma-separated numbers, e.g. RAIN_PEAK_MMH="0,25"."""
    v = _raw(name)
    x = default if v is None else tuple(float(p) for p in v.replace(";", ",").split(","))
    if n is not None and len(x) != n:
        raise ValueError(f"{name} needs {n} comma-separated numbers, got {v!r}")
    return x


def _check_range(name, x, lo, hi):
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {x}")


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
def unpack_zips(root: Path, dest: Path) -> list[Path]:
    """Extract every .zip under root (read-only mount) into dest; returns the folders."""
    out = []
    for i, z in enumerate(sorted(root.rglob("*.zip")) if root.exists() else []):
        target = (dest / f"zip{i}_{z.stem}").resolve()
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():                     # refuse path traversal
                if not (target / info.filename).resolve().is_relative_to(target):
                    raise ValueError(f"Unsafe entry {info.filename!r} in {z}")
            zf.extractall(target)
        out.append(target)
    return out


def find_files(pattern: str, roots: list[Path]) -> list[Path]:
    return sorted({p for r in roots if r.exists() for p in r.rglob(pattern)
                   if p.is_file() and "__MACOSX" not in p.parts})


def find_dir_with(names: tuple[str, ...], roots: list[Path]) -> Path:
    """The single folder (searched recursively) that contains all of `names`."""
    hits = sorted({p.parent for p in find_files(names[0], roots)
                   if all((p.parent / n).is_file() for n in names)})
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one folder containing {list(names)} under "
            f"{[str(r) for r in roots]} (a dataslot or a previous workflow step); found {hits}")
    return hits[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee_stdout(path: Path):
    """Print as usual (DAFNI log) and also save what is printed to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stdout = old


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
