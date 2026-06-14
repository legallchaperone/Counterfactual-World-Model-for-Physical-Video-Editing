#!/usr/bin/env python3
"""DEPRECATED shim — the remove interface now lives in tools/e2w_remove.py.

This module re-exports the remove interface for back-compat while references
(orchestrator path calls, docs) are migrated to the new name. It will be removed
once migration is complete. Do not add new code here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2w_remove import *  # noqa: F401,F403,E402
from e2w_remove import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
