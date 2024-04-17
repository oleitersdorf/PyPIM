import pypim as pim
import unittest
import numpy as np
import math
from parameterized import parameterized
import utils

class ReductionTests(unittest.TestCase):
    @parameterized.expand([
       ('sum', np.sum, np.dtype(np.int32)),
       ('prod', np.prod, np.dtype(np.int32)),
       ('sum', np.sum, np.dtype(np.float32)),
       ('prod', np.prod, np.dtype(np.float32))
    ]) 
    def test_reduce(self, function, gt_func, dtype):
        print(f'Testing reduction with {function} on type {dtype}')
        nelem = 2 ** 16
        ninputs = 1
        
        # Allocate and initialize the tensors
        refs = [utils.rand(nelem, dtype) for _ in range(ninputs)]
        tensors = [pim.from_numpy(x) for x in refs]
        
        # Perform the arithmetic function
        with pim.Profiler():
            result = getattr(tensors[0], function)()
        result = dtype.type(result)

        # Compare to ground-truth
        ground_truth = gt_func(refs[0]).astype(dtype)
        np.testing.assert_allclose(ground_truth, result, atol=1e-4)

if __name__ == '__main__':
    unittest.main()
