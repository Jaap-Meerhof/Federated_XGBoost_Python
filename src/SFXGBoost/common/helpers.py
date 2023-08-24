import numpy as np

def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, (list, np.ndarray)):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened