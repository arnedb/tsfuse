import warnings
import numpy as np


def to_dataframe(result):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pandas as pd

        def collection_to_dataframe(n, x):
            n = str(n).replace('[', '(').replace(']', ')')
            df = pd.DataFrame()
            values = x.values
            if isinstance(x.shape[1], tuple) or (x.shape[1] > 1):
                pass
                # for d in range(x.shape[2]):
                #     for t in range(x.shape[1]):
                #         df['{}[{}, {}]'.format(n, t, d).replace('[', '{').replace(']', '}')] = values[:, t, d]
            elif x.shape[2] > 1:
                for d in range(x.shape[2]):
                    df['{}[{}]'.format(n, d).replace('[', '{').replace(']', '}')] = values[:, 0, d]
            else:
                df[n] = values[:, 0, 0]
            return df

        assert len(set(len(result[node]) for node in result if (result[node] is not None)
            and (not isinstance(result[node].shape[1], tuple)))) <= 1
        
        dfs = []

        for node in result:
            output = result[node]
            if output is not None:
                if isinstance(output, (list, np.ndarray)):
                    for i in range(len(output)):
                        dfs.append(collection_to_dataframe(node, output[i]))
                else:
                    dfs.append(collection_to_dataframe(node, output))

    df = pd.concat(dfs, axis=1)
    df = df.loc[:, sorted(df.columns)]

    return df
