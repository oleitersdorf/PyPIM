import pypim as pim
import unittest
import numpy as np
import math
from parameterized import parameterized
import utils

class UnitTests(unittest.TestCase):
    @parameterized.expand([
       ('__add__', np.add, np.dtype(np.int32)),
       ('__sub__', np.subtract, np.dtype(np.int32)),
       ('__mul__', np.multiply, np.dtype(np.int32)),
       ('__truediv__', np.true_divide, np.dtype(np.int32)),
       ('__add__', np.add, np.dtype(np.float32)),
       ('__sub__', np.subtract, np.dtype(np.float32)),
       ('__mul__', np.multiply, np.dtype(np.float32)),
       ('__truediv__', np.true_divide, np.dtype(np.float32))
    ]) 
    def test_arit(self, function, gt_func, dtype):
        print(f'Testing arithmetic with {function} on type {dtype}')
        nelem = 2 ** 16
        ninputs = 2
        
        # Allocate and initialize the tensors
        refs = [utils.rand(nelem, dtype) for _ in range(ninputs)]
        tensors = [pim.from_numpy(x) for x in refs]
        
        # Perform the arithmetic function
        with pim.Profiler():
            result = getattr(tensors[0], function)(tensors[1])
        result = pim.to_numpy(result)

        # Compare to ground-truth
        ground_truth = gt_func(refs[0], refs[1]).astype(dtype)
        np.testing.assert_array_equal(ground_truth, result)

    @parameterized.expand([
       ('__lt__', np.less, np.dtype(np.int32)),
       ('__le__', np.less_equal, np.dtype(np.int32)),
       ('__gt__', np.greater, np.dtype(np.int32)),
       ('__ge__', np.greater_equal, np.dtype(np.int32)),
       ('__eq__', np.equal, np.dtype(np.int32)),
       ('__ne__', np.not_equal, np.dtype(np.int32)),
       ('__lt__', np.less, np.dtype(np.float32)),
       ('__le__', np.less_equal, np.dtype(np.float32)),
       ('__gt__', np.greater, np.dtype(np.float32)),
       ('__ge__', np.greater_equal, np.dtype(np.float32)),
       ('__eq__', np.equal, np.dtype(np.float32)),
       ('__ne__', np.not_equal, np.dtype(np.float32))
    ]) 
    def test_comp(self, function, gt_func, dtype):
        print(f'Testing comparison with {function} on type {dtype}')
        nelem = 2 ** 16
        ninputs = 2
        
        # Allocate and initialize the tensors
        refs = [utils.rand(nelem, dtype) for _ in range(ninputs)]
        tensors = [pim.from_numpy(x) for x in refs]
        
        # Perform the arithmetic function
        with pim.Profiler():
            result = getattr(tensors[0], function)(tensors[1])
        result = pim.to_numpy(result)

        # Compare to ground-truth
        ground_truth = gt_func(refs[0], refs[1]).astype(bool)
        np.testing.assert_array_equal(ground_truth, result)

    @parameterized.expand([
       ('sin', np.sin, np.dtype(np.float32)),
       ('cos', np.cos, np.dtype(np.float32)),
    ]) 
    def test_cordic(self, function, gt_func, dtype):
        print(f'Testing CORDIC with {function} on type {dtype}')
        nelem = 2 ** 16
        ninputs = 1
        
        # Allocate and initialize the tensors
        refs = [utils.rand(nelem, dtype, -math.pi / 2, math.pi / 2) for _ in range(ninputs)]
        tensors = [pim.from_numpy(x) for x in refs]
        
        # Perform the arithmetic function
        with pim.Profiler():
            result = getattr(pim, function)(tensors[0])
        result = pim.to_numpy(result)

        # Compare to ground-truth
        ground_truth = gt_func(refs[0]).astype(dtype)
        np.testing.assert_allclose(ground_truth, result, atol=1e-5)
        
if __name__ == '__main__':
    unittest.main()
