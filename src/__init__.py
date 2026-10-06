"""Student Performance Classification Package.
GDG ABESEC ML Project.
"""

import sys
from pathlib import Path

# Automatically add local packages directory to sys.path if present
_pkg_dir = Path(__file__).resolve().parent.parent / "packages"
if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

__version__ = "1.0.0"

