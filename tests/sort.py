import pypim as pim
import random
import unittest
from parameterized import parameterized

class TestSort(unittest.TestCase):
   @parameterized.expand([
       (2 ** 2, 2 ** 2, pim.int32),
       (2 ** 4, 2 ** 4, pim.int32),
       (2 ** 6, 2 ** 6, pim.int32),
       (2 ** 8, 2 ** 8, pim.int32),
       (2 ** 10, 2 ** 10, pim.int32),
       (2 ** 12, 2 ** 12, pim.int32),
       (2 ** 14, 2 ** 14, pim.int32),
       (2 ** 16, 2 ** 16, pim.int32),
       (2 ** 18, 2 ** 18, pim.int32),
       (2 ** 20, 2 ** 20, pim.int32),
       (2 ** 22, 2 ** 22, pim.int32),
       (2 ** 24, 2 ** 24, pim.int32),
       (2 ** 26, 2 ** 26, pim.int32),
   ])
   def test_sort(self, nelem, group_size, dtype):
        print(f'Testing sort with {nelem} split into groups of {group_size} and {dtype}')
        assert((nelem % group_size) == 0)
        
        # Allocate and initialize the tensor
        x = pim.Tensor(nelem, dtype=dtype)
        numbers = [set()] * (nelem // group_size)
        for i in range(nelem):
            val = random.randint(-2 ** 30, 2 ** 30 - 1)
            x[i] = val
            numbers[i // group_size].add(val)

        # Perform the sorting algorithm
        with pim.Profiler():
            y = x.sort(group_size)

        # Verify sorted and identical elements
        for i in range(group_size - 1):
            self.assertTrue(y[i] <= y[i + 1])
        for i in range(nelem):
            self.assertTrue(y[i] in numbers[i // group_size])
        
if __name__ == '__main__':
    unittest.main()
