import pytest
import numpy as np

from tsfuse.data.synthetic import brownian
from tsfuse.computation import Graph, Input
from tsfuse.transformers import Add, Mean, Variance, PowerSpectralDensity


@pytest.fixture
def graph():
    return Graph([
        Mean(Input('x')),
        PowerSpectralDensity(Input('y')),
        Variance(Add(Input('x'), Input('y'))),
    ])


def test_add_node_input_node_new():
    graph = Graph()
    node = Input(0)
    graph.add_node(node)
    assert len(graph.inputs) == 1
    assert graph.inputs[0] == node


def test_add_node_input_node_existing():
    graph = Graph()
    node = Input(0)
    graph.add_node(node)
    graph.add_node(Input(0))
    assert len(graph.inputs) == 1
    assert graph.inputs[0] == node


def test_add_node_input_id_new_int():
    graph = Graph()
    graph.add_node(0)
    assert len(graph.inputs) == 1
    assert graph.inputs[0].input_id == 0


def test_add_node_input_id_existing_int():
    graph = Graph()
    node = Input(0)
    graph.add_node(node)
    graph.add_node(0)
    assert len(graph.inputs) == 1
    assert graph.inputs[0] == node


def test_add_node_input_id_new_str():
    graph = Graph()
    graph.add_node('x')
    assert len(graph.inputs) == 1
    assert graph.inputs['x'].input_id == 'x'


def test_add_node_input_id_existing_str():
    graph = Graph()
    node = Input('x')
    graph.add_node(node)
    graph.add_node('x')
    assert len(graph.inputs) == 1
    assert graph.inputs['x'] == node


def test_add_node_transformer():
    graph = Graph()
    graph.add_node(Mean(Input(0)))
    graph.add_node(Variance(Input(0)))
    assert len(graph.nodes) == 3
    assert len(graph.inputs) == 1
    assert len(graph.outputs) == 2


def test_add_node_transformer_with_optimization():
    graph = Graph()
    graph.add_node(Mean(Input(0)), optimize=True)
    graph.add_node(Variance(Input(0)), optimize=True)
    assert len(graph.nodes) == 4
    assert len(graph.inputs) == 1
    assert len(graph.outputs) == 2


def test_transform(graph):
    x = brownian()
    y = brownian()
    result = graph.transform({'x': x, 'y': y}, return_dataframe=False)
    assert len(result) == 3
    np.testing.assert_almost_equal(
        result[graph.outputs[0]].values,
        Mean().transform(x).values
    )
    np.testing.assert_almost_equal(
        result[graph.outputs[1]].values,
        PowerSpectralDensity().transform(y).values
    )
    np.testing.assert_almost_equal(
        result[graph.outputs[2]].values,
        Variance().transform(Add().transform(x, y)).values
    )


def test_transform_to_dataframe(graph):
    x = brownian()
    y = brownian()
    result = graph.transform({'x': x, 'y': y}, return_dataframe=True)
    print(graph.outputs)
    assert result.shape == (10, 4)
