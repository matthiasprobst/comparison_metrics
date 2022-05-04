import setuptools

from comparson_metrics._version import __version__

name = 'comparison_metrics'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=name,
    version=__version__,
    author="Matthias Probst",
    author_email="matthias.probst@kit.edu",
    description="Collection of comparison metrics mainly intended for vector field comparisons.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatthiasProbst/comparison_metrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'appdirs',
        'numpy>=1.19',
        'bibtexparser',
        'appdirs',
        'xarray',
        'pathlib',
        'bibtexparser',
        'xarray'
        'pint_xarray',
        'pytest'
    ],
)
