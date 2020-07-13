
Tuberculosis Portal Chest X-Ray Outlier Detector
++++++++++++++++++++++++++++++++++++++++++++++++

TODO

Documentation
-------------

TODO

Installation
------------

Download wheel from `Github Actions`_ under the latest master build in the
"python-package" artifacts. Then install::

 python -m pip install tbpcxr-py3-none-any.whl

SITK-IBEX requires SimpleITK version 2.0 which is still under development.
For now, use the `latest`_ development version of SimpleITK::

    python -m pip install --upgrade --pre SimpleITK --find-links https://github.com/SimpleITK/SimpleITK/releases/tag/latest

Other dependencies are conventionally specified in `setup.py` and `requirements.txt`.


Example
-------

TODO





Contact
-------

Please use the `GitHub Issues`_ for support and code issues related to the tbp-cxr-outlier project.

Additionally, we can be emailed at: bioinforamtics@niaid.nih.gov Please include "tbp-cxr-outlier" in the subject line.


.. _SimpleITK toolkit: https://simpleitk.org
.. _pip: https://pip.pypa.io/en/stable/quickstart/
.. _Github Actions: https://github.com/niaid/tbp-cxr-outlier/actions?query=branch%3Amaster
.. _latest: https://github.com/SimpleITK/SimpleITK/releases
.. _GitHub Issues:  https://github.com/niaid/tbp-cxr-outlier
