"""HearthLM: a small decoder-only language model with a complete training stack."""

from .config import HearthConfig, PRESETS, build_config
from .model import HearthLM

__version__ = "0.1.0"

__all__ = ["HearthConfig", "PRESETS", "build_config", "HearthLM", "__version__"]
