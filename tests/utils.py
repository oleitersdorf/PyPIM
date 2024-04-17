import numpy as np

def rand(size, dtype, low=None, high=None):
    rng = np.random.default_rng()

    if dtype == np.dtype(np.int32):

        low = -2 ** 30 if low is None else low
        high = 2 ** 30 if high is None else high

        return rng.integers(low, high, size=size, dtype=dtype)
    
    elif dtype == np.dtype(np.float32):

        if low is None and high is None:
            return rng.standard_normal(size=size, dtype=dtype)

        elif low is not None and high is not None:
            return rng.uniform(size=size, low=low, high=high).astype(dtype)
        
    raise NotImplementedError()
