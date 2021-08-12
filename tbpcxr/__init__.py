import pkg_resources as res
import os

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version(__name__)
except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        # package is not installed
        pass
except PackageNotFoundError:
    # package is not installed
    pass


model_list = [os.path.splitext(fn)[0] for fn in res.resource_listdir(__name__, "model")]

del os

__author__ = ["Bradley Lowekamp"]
