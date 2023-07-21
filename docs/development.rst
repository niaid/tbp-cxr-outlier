Development
===========

To setup for development:

.. code-block:: bash

  python -m pip install --editable .[dev]

Git LFS
=======

Git `Large File Storage <https://git-lfs.github.com>`_ (LFS) is used to store larger files in the repository such as
test images, trained models, and other data ( i.e. not text based code ). Before the repository is cloned, git lfs must
be installed on the system and set up on the users account. The `tool's documentation <https://git-lfs.github.com>`_
provides details on installation, set up, and usage that is not duplicated here. Once set up the git usage is usually
transparent with operation such as cloning, adding files, and changing branches.

The ".gitattributes" configuration file automatically places files in the directories "test/data" and "my_pkg/data" to
be stored in Git LFS.


Linting
=======

The linting processes are configured and run with `pre-commit <https://pre-commit.com>`_. Using pre-commit provides
a single file ( ".pre-commit-config.yaml" ) configuration for both execution of CI and local git pre-commit hooks. The
"pre-commit" package does not need to be installed in the projects venv. Once initialized for the project, pre-commit
will manage the versions of the tools in a separate environment, that is automatically managed.

The following is the `quick start guide <https://pre-commit.com/#quick-start>`_.

The linting process uses both `Black <https://black.readthedocs.io/en/stable/>`_  and
`flake8 <https://flake8.pycqa.org/en/latest/>`_ to ensure uncompromising code formatting and identify programmatic
problems. The black code formatting tool must be used to auto format new code before committing:

.. code:: bash

    python -m black .