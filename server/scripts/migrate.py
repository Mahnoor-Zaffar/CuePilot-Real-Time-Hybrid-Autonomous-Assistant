#!/usr/bin/env python
"""Apply SQL migrations without booting the gateway.

    cd server && .venv/bin/python scripts/migrate.py
    cd server && .venv/bin/python scripts/migrate.py --dim 256

The gateway also migrates on boot; this exists for operators who want to run
schema changes as their own step (CI, a job, a pre-deploy hook).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infra.migrations import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
