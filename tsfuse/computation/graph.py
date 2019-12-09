from graphviz import Digraph

from .nodes import Node, Input, Constant, Transformer
from .apply import compute
from .util import to_dataframe


class Graph(object):
    """
    Computation graph.

    A computation graph is a directed acyclic graph (DAG) whose nodes are connected by edges that
    represent the flow of the data. There are three types of nodes:

    - **Inputs:** placeholders for the input time series data.
    - **Constants:** values that do not depend on the input data.
    - **Transformers:** computation steps which create new data from existing data. The input of a
      transformer is given by its parent nodes. See :ref:`transformers` for an overview of all
      available transformers.

    The outputs of a computation graph are the values computed by all transformers that either
    have no outgoing edges, or are marked as an output node.

    Parameters
    ----------
    nodes : list(Node), optional
        Initialize the computation graph with a list of nodes.
        Nodes can also be added later using :meth:`~tsfuse.computation.Graph.add_node`

    Attributes
    ----------
    nodes : list(Node)
        All nodes of the graph.
    inputs : dict(str, Input)
        Input nodes, given as a dictionary of ``{input_id: node}`` items.
    constants : list(Constant)
        Constant nodes.
    transformers : list(Transformer)
        Transformer nodes.
    outputs : list(Node)
        Output nodes.
    """

    def __init__(self, nodes=None):
        self._nodes = []
        self._inputs = dict()
        self._parents = dict()
        self._children = dict()
        self._traces = dict()
        if nodes is not None:
            if isinstance(nodes, Node):
                self.add_node(nodes)
            else:
                for node in nodes:
                    self.add_node(node)

    @property
    def nodes(self):
        return self._nodes

    @property
    def inputs(self):
        return {n.input_id: n for n in self.nodes if isinstance(n, Input)}

    @property
    def constants(self):
        return [n for n in self.nodes if isinstance(n, Constant)]

    @property
    def transformers(self):
        return [n for n in self.nodes if isinstance(n, Transformer)]

    @property
    def parents(self):
        return self._parents

    @property
    def children(self):
        return self._children

    @property
    def traces(self):
        return self._traces

    @property
    def outputs(self):
        return [n for n in self.nodes if n.is_output]

    def add_node(self, node, optimize=True):
        """
        Add a node to the computation graph.

        Parameters
        ----------
        node : Node
            Node that will be added to the graph.
        optimize : boolean, optional
            Merge equal paths to speed up the computation by avoiding to recompute the same
            transformations more than once. Default: True

        Returns
        -------
        node : Node
        """
        if isinstance(node, (Input, int, str)):
            return self._add_input(node)
        elif isinstance(node, Transformer) and hasattr(node, 'graph'):
            return self._add_graph_transformer(node, optimize=optimize)
        elif node not in self._nodes:
            if optimize and node.trace in self.traces:
                return self.traces[node.trace]
            else:
                return self._add_node(node, optimize=optimize)
        else:
            return node

    def transform(self, X, return_dataframe=True, chunk_size=None, n_jobs=None):
        """
        Compute all outputs of the graph.

        Parameters
        ----------
        X : dict(int or str: Collection)
            Data collections used as inputs for the graph. Collection ``X[i]`` will
            be used for ``graph.inputs[i]``.
        return_dataframe : bool, default True
            Return the graph's output as a pandas DataFrame.
        chunk_size : int, optional
            Split the input data collections into chunks ``c``
            with ``c.shape[0] == chunk_size`` and process each block separately.
        n_jobs : int, optional
            Number of chunks to process simultaneously,
            only relevant if a chunk size is specified.
        """
        output = compute(self, X, chunk_size=chunk_size, n_jobs=n_jobs)
        if return_dataframe:
            return to_dataframe(output)
        else:
            return output

    def to_dot(self):
        """
        Visualize the computation graph.
        """
        dot = Digraph()
        for node in self._nodes:
            dot.node(str(node.trace), node.name)
        for node in self._nodes:
            for parent in self._parents[node]:
                dot.edge(str(parent.trace), str(node.trace))
        return dot

    def _add_input(self, node):
        if isinstance(node, Input) and node.input_id not in self.inputs:
            self._inputs[node.input_id] = node
            self._nodes.append(node)
            self._traces[node.trace] = node
            self._children[node] = []
            self._parents[node] = []
            return node
        elif isinstance(node, Input):
            return self._inputs[node.input_id]
        elif isinstance(node, (int, str)) and node not in self.inputs:
            node = Input(node)
            self._inputs[node.input_id] = node
            self._nodes.append(node)
            self._traces[node.trace] = node
            self._children[node] = []
            self._parents[node] = []
            return node
        elif isinstance(node, (int, str)):
            return self._inputs[node]

    def _add_node(self, node, optimize=True):
        self._traces[node.trace] = node
        self._parents[node] = []
        self._children[node] = []
        for parent in node.parents:
            parent = self.add_node(parent, optimize=optimize)
            self._children[parent].append(node)
            self._parents[node].append(parent)
        self._nodes.append(node)
        return node

    def _add_graph_transformer(self, node, optimize=True):
        # Create graph
        graph = node.graph(*[Input(i) for i in range(len(node.parents))])
        # Change inputs
        for i, parent in enumerate(node.parents):
            for n in graph.inputs[i].children:
                n._parents = [
                    parent if isinstance(p, Input) and p.input_id == i else p
                    for p in n._parents
                ]
        # Add output node
        output = graph.outputs[-1]
        output._is_output = node.is_output
        output = self.add_node(output, optimize=optimize)
        return output

    def _repr_svg_(self):
        return self.to_dot()._repr_svg_()
