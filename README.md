<h1 align="center">TSFuse</h1>

<p align="center">Python package for automatically constructing features from multiple time series</p>

<p align="center">
    <a href="https://badge.fury.io/py/tsfuse">
        <img alt="PyPI" src="https://badge.fury.io/py/tsfuse.svg">
    </a>
    <a href="https://github.com/arnedb/tsfuse/actions/workflows/tests.yml">
        <img alt="tests" src="https://github.com/arnedb/tsfuse/workflows/tests/badge.svg" />
    </a>
</p>

<hr>

## Installation

TSFuse requires Python 3, NumPy, and Cython:

    pip install numpy cython
    
The latest release can be installed using pip:

    pip install tsfuse
    
## Quickstart

The example below shows the basic usage of TSFuse.

### Data format

The input of TSFuse is a dataset where each instance is a window that consists of multiple time series and a label.

#### Time series

Time series are represented using a dictionary where each entry represents a univariate or multivariate time series. As an example, let's create a dictionary with two univariate time series:

```python
from pandas import DataFrame
from tsfuse.data import Collection
X = {
    "x1": Collection(pd.DataFrame({
        'id':   [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'time': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        'data': [1, 2, 3, 1, 2, 3, 3, 2, 1, 3, 2, 1],
    })),
    "x2": Collection(pd.DataFrame({
        'id':   [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'time': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        'data': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    })),
}
```

The two univariate time series are named `x1` and `x2` and each series is represented as a `Collection` object. Each ``Collection`` is initialized with a DataFrame that has three columns:

- `id` which is the identifier of each instance, i.e., each window,
- `time` which contains the time stamps,
- `data` contains the time series data itself.

For multivariate time series data, there can be multiple columns similar to the `data` column. For example, the data of a tri-axial accelerometer would have three columns `x`, `y`, `z` instead of `data` as it simultaneously measures the `x`, `y`, `z` acceleration.

#### Labels

There should be one target value for each window, so we create a `Series` where the index contains all unique `id` values of the time series data and the data consists of the labels:

```python
from pandas import Series
y = Series(index=[0, 1, 2, 3], data=[0, 0, 1, 1])
```

### Feature construction

To construct features, TSFuse provides a `construct` function which takes time series data `X` and target data `y` as input, and returns a `DataFrame` where each column corresponds to a feature. In addition, this function can return a computation graph which contains all transformation steps required to compute the features for new data:

```python
from tsfuse import construct
features, graph = construct(X, y, transformers="minimal", return_graph=True)
```

To apply this computation graph to new data, simply call `transform` with a time series dictionary `X` as input:

```python
features = graph.transform(X)
```
    
## Documentation

The documentation is available on [https://arnedb.github.io/tsfuse/](https://arnedb.github.io/tsfuse/)

## Citing TSFuse

If you use TSFuse for a scientific publication, please consider citing this paper:

> De Brabandere, A., Op De Beéck, T., Hendrickx, K., Meert, W., & Davis, J. [TSFuse: automated feature construction for multiple time series data](https://doi.org/10.1007/s10994-021-06096-2). *Machine Learning* (2022)

```bibtex
@article{tsfuse,
    author  = {De Brabandere, Arne
               and Op De Be{\'e}ck, Tim
               and Hendrickx, Kilian
               and Meert, Wannes
               and Davis, Jesse},
    title   = {TSFuse: automated feature construction for multiple time series data},
    journal = {Machine Learning},
    year    = {2022},
    doi     = {10.1007/s10994-021-06096-2},
    url     = {https://doi.org/10.1007/s10994-021-06096-2}
}
```
