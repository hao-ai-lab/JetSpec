"""Block geometry, independent of environment parsing and serving layout."""
from dataclasses import dataclass


# Narrow blocks retain the storage originally sized for B16.
MIN_WINDOW_ROWS = 16


@dataclass(frozen=True)
class Geometry:
    block: int

    @classmethod
    def from_drafts(cls, drafts):
        return cls(drafts + 1)

    @property
    def drafts(self):
        return self.block - 1

    @property
    def window_rows(self):
        return max(MIN_WINDOW_ROWS, self.block)

    @property
    def served(self):
        """Supported blocks, including the explicit legacy fourteen-draft profile."""
        return self.block in (4, 8, 15, 16, 32)
