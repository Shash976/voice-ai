"""validate.py — gate 1 of the cascade: cheap, tool-free config validation.

Two checks, both instant (microseconds), run before any tool touches the config:

  1. Membership   — every parameter's value is one of its declared `choices`.
  2. Constraints  — every declarative boolean expression in the search space's
                    `constraints:` list evaluates True for this config.

Constraints are plain Python boolean expressions over the parameter names, e.g.
    "not (core_utilization >= 70 and place_density >= 0.75)"
They are eval'd in a RESTRICTED namespace: no builtins, only the config values.
This keeps validity rules DATA-DRIVEN — add a line to the YAML and the funnel
enforces it, no code change.

Returns (ok: bool, reason: str). `reason` is "" on success, else the first
failing check (a missing/illegal value, or the constraint expression that failed).
"""

from __future__ import annotations


def _membership_error(config: dict, space: dict) -> str:
    for name, spec in space.items():
        if name not in config:
            return f"missing parameter: {name}"
        choices = spec.get("choices")
        if choices is not None and config[name] not in choices:
            return f"{name}={config[name]!r} not in choices {choices}"
    # flag stray params not in the space (typo / stale agent state)
    for name in config:
        if name not in space:
            return f"unknown parameter: {name}"
    return ""


def _constraint_error(config: dict, constraints: list[str]) -> str:
    # Restricted eval: expose ONLY the config values, no builtins. Constraint
    # expressions are author-controlled (from the search-space YAML), not user
    # input, but we still sandbox to fail loudly on anything unexpected.
    safe_globals = {"__builtins__": {}}
    for expr in constraints or []:
        try:
            ok = bool(eval(expr, safe_globals, dict(config)))  # noqa: S307 (sandboxed)
        except Exception as exc:  # noqa: BLE001 — a broken rule should fail closed
            return f"constraint error in {expr!r}: {exc}"
        if not ok:
            return f"constraint failed: {expr}"
    return ""


def validate(config: dict, space: dict, constraints: list[str] | None = None) -> tuple[bool, str]:
    """Return (is_valid, reason). reason is '' when valid."""
    err = _membership_error(config, space)
    if err:
        return False, err
    err = _constraint_error(config, constraints or [])
    if err:
        return False, err
    return True, ""
