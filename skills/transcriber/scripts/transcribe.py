#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.cli import main
from audio_transcription.exiting import run_entry_point


if __name__ == "__main__":
    run_entry_point(main)
