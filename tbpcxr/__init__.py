from .data import model_list

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["model_list"]
__author__ = ["Bradley Lowekamp"]
