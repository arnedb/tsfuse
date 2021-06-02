import time
import warnings
from multiprocessing import Pool

__all__ = [
    'compute',
]


def compute(graph, inputs, chunk_size=None, n_jobs=None):
    """
    Compute the outputs of a computation graph.

    Parameters
    ----------
    graph : Graph
    inputs : dict
    chunk_size : int, optional
    n_jobs : int, optional

    Returns
    -------
    outputs : dict
    """
    n = inputs[list(inputs)[0]].shape[0]
    if not all(inputs[i].shape[0] == n for i in inputs):
        raise ValueError("Size of axis 0 must be equal for all inputs")
    if graph.optimized:
        graph_original = graph
        graph = graph.optimized
    if chunk_size is None:
        data = {}
        for input_id in graph.inputs:
            data[graph.inputs[input_id]] = inputs[input_id]
        for n in graph.constants:
            data[n] = n.output
        for node in graph.nodes:
            if node not in data:
                data_parents = [data[n] for n in graph.parents[node]]
                t = time.time()
                try:
                    data[node] = node.transform(*data_parents)
                except Exception:  # TODO: Make exception more specific
                    data[node] = None
                    warnings.warn('Error for {}'.format(node))
        if graph_original.optimized:
            return {graph.original[node]: data[node] for node in graph.outputs}
        else:
            return {node: data[node] for node in graph.outputs}
    else:
        if n_jobs is None:
            return _compute_chunks(inputs, graph_original, chunk_size)
        else:
            return _compute_chunks_parallel(inputs, graph_original, chunk_size, n_jobs)


def _compute_chunks(inputs, graph, chunk_size):
    n = next(iter(inputs.values())).shape[0] if isinstance(inputs, dict) else inputs[0].shape[0]
    outputs = None
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        result_chunk = compute(graph, _select(inputs, i, j))
        outputs_chunk = result_chunk
        if outputs is None:
            outputs = outputs_chunk
        else:
            for o in graph.outputs:
                if outputs_chunk[o] is None:
                    outputs[o] = None
                elif outputs[o] is not None:
                    outputs[o] = outputs[o].append(outputs_chunk[o])
    return outputs


def _compute_chunks_parallel(inputs, graph, chunk_size, n_jobs):
    pool = Pool(processes=n_jobs)
    n = next(iter(inputs.values())).shape[0] if isinstance(inputs, dict) else inputs[0].shape[0]
    outputs_chunks = [
        pool.apply(compute, args=(graph, _select(inputs, i, min(i + chunk_size, n))))
        for i in range(0, n, chunk_size)
    ]
    outputs = None
    for i in range(len(outputs_chunks)):
        outputs_chunk = outputs_chunks[i]
        if outputs is None:
            outputs = outputs_chunk[:]
        else:
            for o in range(len(outputs)):
                if outputs_chunk[o] is None:
                    outputs[o] = None
                else:
                    outputs[o] = outputs[o].append(outputs_chunk[o], axis=0)
        outputs_chunks[i] = None
    return outputs


def _select(inputs, i, j):
    selection = {} if isinstance(inputs, dict) else [None] * len(inputs)
    indices = inputs if isinstance(inputs, dict) else range(len(inputs))
    for ind in indices:
        selection[ind] = inputs[ind].iloc[i:j, :, :]
    return selection
