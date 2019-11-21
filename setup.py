from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='tsfuse',
    version='1.0dev',
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    setup_requires=[
        'setuptools',
        'Cython>=0.28.5',
        'numpy>=1.16.1'
    ],
    install_requires=[
        'six>=1.12.0',
        'graphviz>=0.10.1',
        'scipy>=1.2.1',
        'scikit-learn>=0.20.1',
        'statsmodels>=0.9.0',
        'Pint>=0.9',
    ],
    extras_require={'test': [
        'pytest',
        'pandas>=0.24.2'
    ]},
    ext_modules=cythonize([
        Extension(
            'tsfuse.data.df', ['tsfuse/data/df.pyx'],
            include_dirs=[np.get_include()]
        ),
        Extension(
            'tsfuse.transformers.calculators.*', ['tsfuse/transformers/calculators/*.pyx'],
            include_dirs=[np.get_include()]
        ),
    ]),
)
