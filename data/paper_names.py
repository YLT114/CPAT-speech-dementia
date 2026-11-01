"""Public paper names for datasets with legacy storage identifiers.

The project keeps the internal identifiers because they are part of existing
directory layouts and cached artifacts. Use :func:`paper_dataset_name` only
for logs, tables, CSV/JSON metadata, and other public-facing output.
"""

from typing import Any, Dict


PAPER_DATASET_NAMES: Dict[str, str] = {
    "ADReSS-IS2020": "ADReSS",
    "ADReSSo21": "ADReSSo",
    "Pitt": "DB",
}


def paper_dataset_name(internal_name: Any) -> str:
    """Return the paper-facing dataset name without changing storage keys."""
    name = str(internal_name)
    return PAPER_DATASET_NAMES.get(name, name)


__all__ = ["PAPER_DATASET_NAMES", "paper_dataset_name"]
