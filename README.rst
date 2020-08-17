
Tuberculosis Portal Chest X-Ray Outlier Detector
++++++++++++++++++++++++++++++++++++++++++++++++

The tbpcxr Python module provides tools to detect outliers of chest x-ray (CXR) images. Machine learning or artificial
intelligence (AI) methods are used to train a model representing "normal" CXR images, and then new images are
classified as near the (normal) model,  or significantly different from the model and therefore an "outlier".

The following steps are used in the algorithm to detect outliers:

 - Normalize the image into a standard size and intensity range.
 - Register the image to a CXR atlas.
 - Transform the image into an approximations with a precomputed principle component basis.
 - Predict if the image is an outlier determined by how well the image fits to the model.


Documentation
-------------

The published Sphinx documentation is available here: https://niaid.github.io/tbp-cxr-outlier/

The master built Sphinx documentation is available for download from
`Github Actions`_ under the build as "sphinx-docs".

Installation
------------

The Python module is distributed as a `wheel`_ binary package. Download the latest tagged release from the
`Github Releases`_ page. Then install::

 python -m pip install tbpcxr-0.1-py3-none-any.whl

Wheels from the master branch can be download wheel from `Github Actions`_ in the
"python-package" artifact.

The `tbpcxr` module requires SimpleITK version 2.0 which is still under development.
For now, use the `latest`_ development version of SimpleITK::

 python -m pip install --upgrade --pre SimpleITK --find-links https://github.com/SimpleITK/SimpleITK/releases/tag/latest

Other dependencies are conventionally specified in `setup.py` and `requirements.txt` and therefore installed as dependencies when
the wheel is installed.


Contact
-------

Please use the `GitHub Issues`_ for support and code issues related to the tbp-cxr-outlier project.

Additionally, we can be emailed at: bioinformatics@niaid.nih.gov Please include "tbp-cxr-outlier" in the subject line.


.. _SimpleITK toolkit: https://simpleitk.org
.. _pip: https://pip.pypa.io/en/stable/quickstart/
.. _Github Actions: https://github.com/niaid/tbp-cxr-outlier/actions?query=branch%3Amaster
.. _latest: https://github.com/SimpleITK/SimpleITK/releases
.. _GitHub Issues:  https://github.com/niaid/tbp-cxr-outlier
.. _wheel: https://www.python.org/dev/peps/pep-0427/
.. _Github Releases: https://github.com/niaid/tbp-cxr-outlier/releases
