"""
conftest.py — Shared pytest fixtures.
"""

import sys
import os
from pathlib import Path

import pytest

# Add Separated Pipelines root to sys.path so imports work
_SP_ROOT = Path(__file__).resolve().parent.parent
if str(_SP_ROOT) not in sys.path:
    sys.path.insert(0, str(_SP_ROOT))
