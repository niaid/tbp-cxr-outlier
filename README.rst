
Tuberculosis Portal Chest X-Ray Outlier Detector
++++++++++++++++++++++++++++++++++++++++++++++++

The tbpcxr Python module provides tools to detect outliers of chest x-ray (CXR) images. Machine learning or artificial
intelligence (AI) methods are used to train a model representing "normal" CXR images, and then new images are
classified as near the (normal) model,  or significantly different from the model and therefore an "outlier".

The following steps are used in the algorithm to detect outliers:

 - Normalize the image into a standard size and intensity range.
 - Register the image to a CXR atlas.
 - Transform the image into an approximations with a precomputed principal component basis.
 - Predict if the image is an outlier determined by how well the image fits to the model.


Documentation
-------------

The published Sphinx documentation is available here: https://niaid.github.io/tbp-cxr-outlier/

The master built Sphinx documentation is available for download from
`Github Actions`_ under the build as "sphinx-docs".

Installation
------------

The Python module is distributed as a `wheel`_ binary package which defines the package dependencies.

The package `rap_sitkcore`_ is an optional dependency used for reading DICOM files. It provides
additional robust reading for difficult `TBPortals`_ DICOM files which may be addressed in some published
datasets.

- Option 1: Manually Download Wheels from Github Releases

  The wheels can be downloaded from the `Github Releases`_ page and the optional dependency package `rap_sitkcore` can
  be downloaded separately from the `Github rap_sitkcore Releases`_ page. Then the two wheels should be installed together::

     python -m pip tbpcxr-0.5.1-py3-none-any.whl rap_sitkcore-0.5.5-py3-none-any.whl



- Option 2: Use Internal NIAID Artifactory

  It can be specified when installing with the optional dependency package notation `tbpcxr[sitkcore]`. The
  `rap_sitkcore` package is hosted on the NIAID artifactory.

  Internally the `tbpcxr` package is hosted on the NIAID Python Package Index (PyPI) hosted on artifactory and is
  installable with `pip`. This enables internal dependencies to be automatically download from the artifactory. The
  internal repository can be automatically used by setting an environment variable::

     PIP_EXTRA_INDEX_URL

  Then the `tbpcxr` package can be installed::

     python -m pip install tbpcxr[sitkcore]

Github Releases
^^^^^^^^^^^^^^^

Wheels from the master branch can be download wheel from `Github Actions`_ in the "python-package" artifact.

Download the latest tagged release from the `Github Releases`_ page.

The wheel lists the package dependencies which are required for successful installation. This include internal NIAID
packages. If the internal "artifactory" repository is not configured then these additional dependencies will need to be
manually downloaded and installed before install `tbpcxr`. The downloaded wheels can be installed::

 python -m pip install tbpcxr-0.5.1-py3-none-any.whl


Contact
-------

Please use the `GitHub Issues`_ for support and code issues related to the tbp-cxr-outlier project.

Additionally, we can be emailed at: bioinformatics@niaid.nih.gov Please include "tbp-cxr-outlier" in the subject line.

.. _TBPortals: https://tbportals.niaid.nih.gov/
.. _rap_sitkcore: https://github.com/niaid/rap_sitkCore
.. _SimpleITK toolkit: https://simpleitk.org
.. _pip: https://pip.pypa.io/en/stable/quickstart/
.. _Github Actions: https://github.com/niaid/tbp-cxr-outlier/actions?query=branch%3Amaster
.. _GitHub Issues:  https://github.com/niaid/tbp-cxr-outlier
.. _wheel: https://www.python.org/dev/peps/pep-0427/
.. _Github Releases: https://github.com/niaid/tbp-cxr-outlier/releases/latest
.. _Github rap_sitkcore Releases: https://github.com/niaid/rap_sitkCore/releases/latest
