"""Input records and batch collation for SPA and CCA."""

from .cca import CCACollator, CCADataset
from .spa import SPACollator, SPADataset

__all__ = ["SPADataset", "SPACollator", "CCADataset", "CCACollator"]
