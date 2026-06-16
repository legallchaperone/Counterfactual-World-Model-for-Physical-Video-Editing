#!/usr/bin/env python3
"""DEPRECATED shim — the add interface now lives in tools/e2w_add.py.

Re-exports the unified add interface for back-compat while references are migrated.
Do not add new code here; use tools/e2w_add.py directly in docs, runs, tests, and
new automation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2w_add import *  # noqa: F401,F403,E402
from e2w_add import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
