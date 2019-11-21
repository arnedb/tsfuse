import warnings
import numpy as np


def to_dataframe(result):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pandas as pd

        def collection_to_dataframe(n, x):
            df = pd.DataFrame()
            values = x.values
            if x.shape[1] > 1:
                for d in range(x.shape[2]):
                    for t in range(x.shape[1]):
                        df['{}[{}, {}]'.format(str(n), t, d)] = values[:, t, d]
            elif x.shape[2] > 1:
                for d in range(x.shape[2]):
                    df['{}[{}]'.format(str(n), d)] = values[:, 0, d]
            else:
                df[str(n)] = values[:, 0, 0]
            return df

        assert len(set(len(result[node]) for node in result if result[node] is not None)) <= 1
        dfs = []

        for node in result:
            output = result[node]
            if output is not None:
                if isinstance(output.shape[1], (tuple, list, np.ndarray)):
                    for i in range(len(output.shape[1])):
                        dfs.append(collection_to_dataframe(node, output.values[i]))
                else:
                    dfs.append(collection_to_dataframe(node, output))

    return pd.concat(dfs, axis=1)
