"""
Ensure the RNA-FM `fm` package is importable without setting PYTHONPATH manually.

Uses `RNA-FM-main` next to the CRISPR-UniPredict repo (same parent folder as this project).
The duplicate tree under `utils/rna_fm` was removed to avoid two copies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def ensure_rna_fm_import_path(project_root: Optional[Path] = None) -> bool:
    """
    Prepend the directory that contains the `fm` package to sys.path.

    Args:
        project_root: CRISPR-UniPredict root (parent of `utils/`). Resolved from this file if None.

    Returns:
        True if a candidate path was added, False otherwise.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    candidates = [
        project_root.parent / "RNA-FM-main",
    ]

    for base in candidates:
        init_py = base / "fm" / "__init__.py"
        if init_py.is_file():
            root = str(base.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return True
    return False
