#
from setuptools import setup

with open("README.rst", "r") as fp:
    long_description = fp.read()

with open("requirements.txt", "r") as fp:
    requirements = list(filter(bool, (line.strip() for line in fp)))

with open("requirements-dev.txt", "r") as fp:
    dev_requirements = list(filter(bool, (line.strip() for line in fp)))


setup(
    name="tbpcxr",
    use_scm_version={"local_scheme": "dirty-tag"},
    author=["Bradley Lowekamp"],
    author_email="bioinformatics@niaid.nih.gov",
    description="Chest X-Ray outlier detector",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/niaid/tbp-cxr-outlier",
    packages=["tbpcxr"],
    package_data={"tbpcxr": ["model/*.pkl"]},
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=requirements,
    tests_require=dev_requirements,
    setup_requires=["setuptools_scm"],
)
