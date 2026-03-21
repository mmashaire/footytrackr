"""
Root conftest.py — fixes imports for scripts whose filenames start with a digit.

Python cannot import modules whose names begin with a number (e.g. '02_build_value_features_v2').
This file loads such modules via importlib and registers them under their expected import name
so that test files can use `from scripts.build_value_features_v2 import ...` without issue.
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent / "scripts"

_NUMERIC_PREFIX_ALIASES = {
    "scripts.build_value_features_v2": _SCRIPTS_DIR / "02_build_value_features_v2.py",
}

for _module_name, _module_path in _NUMERIC_PREFIX_ALIASES.items():
    if _module_name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_module_name, _module_path)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_module_name] = _mod
        _spec.loader.exec_module(_mod)
