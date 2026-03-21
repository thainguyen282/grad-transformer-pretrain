"""
Compatibility shim: re-export prompt helpers from ``src.utils``.

Use this if you prefer ``from prompter_utils import Prompter`` instead of
``from src.utils import Prompter``. The repo root must be on ``sys.path``
(usually the case when running from project root).
"""

from __future__ import annotations

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.utils import (  # noqa: E402
    Prompter,
    generate_and_tokenize_prompt,
    make_contiguous_,
    report_noncontiguous_params,
)

__all__ = [
    "Prompter",
    "generate_and_tokenize_prompt",
    "make_contiguous_",
    "report_noncontiguous_params",
]
