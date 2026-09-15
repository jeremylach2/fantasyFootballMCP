"""Enforce the dependency direction described in docs/architecture.md.

``domain/`` holds the optimizer, the simulation and the valuation logic. Keeping it free of
I/O is what lets those run in milliseconds under test with no network and no credentials,
so this rule is worth enforcing mechanically rather than by convention.
"""

from __future__ import annotations

import ast
from pathlib import Path

DOMAIN = Path(__file__).resolve().parents[1] / "src" / "ffmcp" / "domain"

FORBIDDEN = {"mcp", "mcp_types", "httpx2", "httpx", "espn_api", "requests", "ffmcp.providers"}


def _imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_domain_layer_performs_no_io() -> None:
    offenders: list[str] = []
    for path in DOMAIN.rglob("*.py"):
        for module in _imported_modules(path.read_text(encoding="utf-8")):
            root = module.split(".")[0]
            if root in FORBIDDEN or module in FORBIDDEN:
                offenders.append(f"{path.name} imports {module}")
    assert not offenders, "domain/ must stay pure: " + "; ".join(offenders)
