import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

os.environ["USE_CYTHONIZE"] = "1"

setup(
    name="tsfuse",
    version="0.1.2",
    description="Automated feature construction for multiple time series data",
    author="Arne De Brabandere",
    project_urls={
        "TSFuse documentation": "https://arnedb.github.io/tsfuse/",
        "TSFuse source": "https://github.com/arnedb/tsfuse",
    },
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    install_requires=[
        "cython",
        "graphviz",
        "matplotlib",
        "numpy",
        "pandas",
        "Pint>=0.9",
        "scikit-learn>=0.20.1",
        "scipy>=1.2.1",
        "statsmodels>=0.12.0",
        "six>=1.12.0",
        "sklearn-gbmi>=1.0.3",
    ],
    extras_require={"test": ["pytest"]},
    ext_modules=cythonize(
        [
            Extension(
                "tsfuse.data.df",
                ["tsfuse/data/df.pyx"],
                include_dirs=[np.get_include()],
            ),
            Extension(
                "tsfuse.transformers.calculators.*",
                ["tsfuse/transformers/calculators/*.pyx"],
                include_dirs=[np.get_include()],
            ),
        ]
    ),
)
