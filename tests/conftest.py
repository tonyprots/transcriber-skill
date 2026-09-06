from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "transcriber"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))
