#!/usr/bin/env python3
"""DEPRECATED shim — the add->remove orchestrator now lives in
tools/e2w_add_then_remove.py. Re-exports it for back-compat while references are
migrated. Do not add new code here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2w_add_then_remove import *  # noqa: F401,F403,E402
from e2w_add_then_remove import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
