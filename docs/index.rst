
=====================================
Welcome to tbpcxr's documentation!
=====================================

.. include:: ../README.rst


Example
-------

.. code-block:: python

 from tbpcxr.model import PCAModel
 from tbpcxr.utilities import normalize_img, read_dcm

 outlier_pcamodel = PCAModel.load_outlier_pcamodel()

 img = read_dcm(path_to_file)

 nimg = normalize_img(img)

 rimg = outlier_pcamodel.register_to_atlas_and_resample(nimg, verbose=0)
 arr = outlier_pcamodel._images_to_arr([rimg])

 if outlier_pcamodel.outlier_predictor(arr)[0] == -1:
    print("{} is an outlier".format(path_to_file))


API Reference
-------------

Documentation for directly using the Python functions.

.. toctree::
   :maxdepth: 2

   api


Command Line Interface Reference
--------------------------------

Documentation for command line usage of the Python module.

.. toctree::
   :maxdepth: 2

   commandline


Development
-----------

The `Git Large File Storage`_ (LFS) is required to retrieve and store images and data files in the tbp-cxr-outlier
repository. The Git LFS client needs to be installed and initialized to automatically perform these operations.

The required packages for development are specified in `requirements-dev.txt`. The tbp-cxr-outlier project must be
install for it to function properly. Specifically, because semantic versioning is done with `setuptools-scm` it must be
installed. To setup for development::

  python -m pip install requirements-dev.txt
  python -m pip install --editable .

New contributions must come from pull requests on GitHub. New features should start as local branch with a name
starting with "feature-" followed by a description. After changes are made check the passing of flake8 and the tests
without warnings or errors::

  python -m flake8
  python -m pytest

Since the repository is internal, the feature branch needs to be
pushed to the *upstream* repository. Next a pull request is made against master, where the CI will automatically run
flake8, pytest and sphinx. When merging the branch with rebased onto the origin, and the feature branch is deleted.

.. _Git Large File Storage : https://git-lfs.github.com