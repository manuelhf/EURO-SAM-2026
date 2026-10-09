"""Check the DAFNI model definitions against the v1beta3 reference.

https://docs.secure.dafni.rl.ac.uk/docs/Reference/model-definition-reference
Also checks that every parameter is read by the model's entry point.

    python dafni/validate_definitions.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MODELS = {"network": "run_network.py", "scenarios": "run_scenarios.py", "gnn": "run_gnn.py"}

META_REQUIRED = ("name", "display_name", "summary", "description", "publisher",
                 "contact_point_name", "contact_point_email")
SUBJECTS = {"Biota", "Boundaries", "Climatology / Meteorology / Atmosphere", "Economy",
            "Elevation", "Environment", "Farming", "Geoscientific Information", "Health",
            "Imagery / Base Maps / Earth Cover", "Inland Waters", "Intelligence / Military",
            "Location", "Oceans", "Planning / Cadastre", "Society", "Structure",
            "Transportation", "Utilities / Communication"}
PARAM_TYPES = {"string", "integer", "number", "boolean", "json", "link"}


def check(model: str, entry: str) -> list[str]:
    path = ROOT / model / "model_definition.yaml"
    d = yaml.safe_load(path.read_text())
    err = []
    if d.get("kind") != "M":
        err.append("kind must be M")
    if d.get("api_version") != "v1beta3":
        err.append("api_version must be v1beta3")
    meta, spec = d.get("metadata") or {}, d.get("spec") or {}
    err += [f"metadata.{k} missing" for k in META_REQUIRED if not meta.get(k)]
    if not re.fullmatch(r"[A-Za-z0-9-]+", str(meta.get("name", ""))):
        err.append("metadata.name: alphanumerics and hyphens only")
    if meta.get("type", "model") not in ("model", "service"):
        err.append("metadata.type must be model or service")
    if "subject" in meta and meta["subject"] not in SUBJECTS:
        err.append(f"metadata.subject {meta['subject']!r} not allowed")
    if bool(meta.get("project_name")) != bool(meta.get("project_url")):
        err.append("project_name and project_url must be given together")

    code = (ROOT / model / entry).read_text()
    names = set()
    for p in (spec.get("inputs") or {}).get("parameters") or []:
        n = p.get("name", "?")
        err += [f"parameter {n}: {k} missing" for k in ("name", "title", "description", "type")
                if k not in p]
        if not isinstance(p.get("required"), bool):
            err.append(f"parameter {n}: required must be true/false")
        if p.get("type") not in PARAM_TYPES:
            err.append(f"parameter {n}: type {p.get('type')!r} not allowed")
        if n in names:
            err.append(f"parameter {n}: duplicate")
        names.add(n)
        if f'"{n}"' not in code:
            err.append(f"parameter {n}: not read by {entry}")
        opts = [o.get("name") for o in p.get("options") or []]
        if opts and "default" in p and str(p["default"]) not in opts:
            err.append(f"parameter {n}: default not among options")
        if "regex" in p and "default" in p and not re.fullmatch(p["regex"], str(p["default"])):
            err.append(f"parameter {n}: default does not match regex")
    for s in (spec.get("inputs") or {}).get("dataslots") or []:
        err += [f"dataslot {s.get('name')}: {k} missing" for k in ("name", "path") if k not in s]
        if not str(s.get("path", "")).startswith("inputs/"):
            err.append(f"dataslot {s.get('name')}: path must be under inputs/")
        if s.get("required") and not s.get("default"):
            err.append(f"dataslot {s.get('name')}: required slots need a default dataset UUID")
    for o in (spec.get("outputs") or {}).get("datasets") or []:
        err += [f"output {o.get('name')}: {k} missing" for k in ("name", "type", "description")
                if k not in o]
    if spec.get("command") and spec["command"][-1] != f"/app/{entry}":
        err.append(f"spec.command should run /app/{entry}")
    return [f"{model}: {e}" for e in err]


if __name__ == "__main__":
    errors = [e for m, entry in MODELS.items() for e in check(m, entry)]
    print("\n".join(errors) or f"OK: {len(MODELS)} model definitions valid")
    sys.exit(1 if errors else 0)
