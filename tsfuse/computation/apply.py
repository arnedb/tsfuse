import time
import warnings
from multiprocessing import Pool

__all__ = [
    'compute',
]


def compute(graph, inputs, chunk_size=None, n_jobs=None, return_timings=False):
    """
    Compute the outputs of a computation graph.

    Parameters
    ----------
    graph : Graph
    inputs : dict
    chunk_size : int, optional
    n_jobs : int, optional
    return_timings : bool, optional

    Returns
    -------
    outputs : dict
    """
    n = inputs[list(inputs)[0]].shape[0]
    if not all(inputs[i].shape[0] == n for i in inputs):
        raise ValueError("Size of axis 0 must be equal for all inputs")
    timings = {node.trace: 0 for node in graph.nodes}
    if chunk_size is None:
        data = {}
        for input_id in graph.inputs:
            data[graph.inputs[input_id].trace] = inputs[input_id]
        for n in graph.constants:
            data[n.trace] = n.output
        for node in graph.nodes:
            if node.trace not in data:
                data_parents = [data[n.trace] for n in graph.parents[node]]
                t = time.time()
                try:
                    data[node.trace] = node.transform(*data_parents)
                except Exception:  # TODO: Make exception more specific
                    data[node.trace] = None
                    warnings.warn('Error for {}'.format(node))
                timings[node.trace] += time.time() - t
        result = [{node: data[node.trace] for node in graph.outputs}]
        if return_timings:
            result.append(timings)
        return result[0] if len(result) == 1 else tuple(result)
    else:
        if n_jobs is None:
            return _compute_chunks(inputs, graph, chunk_size, return_timings)
        else:
            return _compute_chunks_parallel(inputs, graph, chunk_size, n_jobs)


def _compute_chunks(inputs, graph, chunk_size, return_timings):
    n = next(iter(inputs.values())).shape[0] if isinstance(inputs, dict) else inputs[0].shape[0]
    outputs = None
    timings = {node.trace: 0 for node in graph.nodes}
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        result_chunk = compute(graph, _select(inputs, i, j), return_timings=return_timings)
        if return_timings:
            outputs_chunk = result_chunk[0]
            timings_chunk = result_chunk[1]
            timings[i] += timings_chunk[i]
        else:
            outputs_chunk = result_chunk
        if outputs is None:
            outputs = outputs_chunk
        else:
            for o in graph.outputs:
                if outputs_chunk[o] is None:
                    outputs[o] = None
                elif outputs[o] is not None:
                    outputs[o] = outputs[o].append(outputs_chunk[o])
    result = [outputs]
    if return_timings:
        result.append(timings)
    return result[0] if len(result) == 1 else tuple(result)


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
