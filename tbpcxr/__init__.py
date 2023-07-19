import pkg_resources as res
import os

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


model_list = [os.path.splitext(fn)[0] for fn in res.resource_listdir(__name__, "model")]

del os

__author__ = ["Bradley Lowekamp"]
