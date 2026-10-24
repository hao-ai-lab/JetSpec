"""The tools area's declared boundary for reading sealed, retired recipe data.

Measurement receipts and their tests retain these records' original settings.
This helper reads data only; it never imports or executes historical code.
"""
from pathlib import Path


# Adaptation: recipe data lives in the public engine package.
ROOT = Path(__file__).resolve().parents[1]


def recipe_record(name, *, root: Path = ROOT):
    """Return the named serve recipe record without permitting path traversal."""
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError(f'expected a recipe name: {name!r}')
    return root / 'configs' / (name + '.json')
