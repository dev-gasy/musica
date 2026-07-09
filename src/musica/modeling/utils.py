"""Small utility functions for modeling code."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from musica.modeling.constants import QUALITIES, ROOTS


def label_sort_key(label: str) -> int:
    note, quality = label.split("_")
    return ROOTS.index(note) * len(QUALITIES) + QUALITIES.index(quality)


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
