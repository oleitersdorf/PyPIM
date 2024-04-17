import math
from enum import Enum
import numpy as np
import pypim.driver as driver
from . import memory

class Type(Enum):
    """
    Represents the different data types supported.
    """

    int32 = 'int32'
    float32 = 'float32'
    bool = 'bool'

    def __init__(self, x) -> None:
        super().__init__()
        self.x = x

    def __repr__(self):
        return self.x
    
    def __str__(self):
        return self.x
    
int32 = Type.int32
float32 = Type.float32
bool = Type.bool

FROM_NUMPY_DTYPE_CONVERSION = {np.dtype(np.int32): Type.int32, np.dtype(np.float32): Type.float32, np.dtype(np.bool_): Type.bool}
TO_NUMPY_DTYPE_CONVERSION = {Type.int32: np.dtype(np.int32), Type.float32: np.dtype(np.float32), Type.bool: np.dtype(np.bool_)}

class Tensor:
    """
    Represents a multi-dimensional Tensor.
    """

    def __init__(self, *shape, dtype: Type = Type.int32, ref_tensor = None, default_value = None, addr = None):
        """
        Initializes the tensor with the given dimensions and data type
        """

        if len(shape) == 0 or math.prod(shape) == 0:
            raise RuntimeError("Empty shape provided.")
        self.shape = shape
        self.dim = len(shape)
        self.stride = tuple((math.prod(shape[1 + x:]) for x in range(self.dim)))
        self.nelem = math.prod(shape)
        self.dtype = dtype

        # Allocate memory for the tensor
        if addr is None:
            self.addr = memory.malloc(self.nelem, ref_addr=(None if ref_tensor is None else ref_tensor.addr))
        else:
            self.addr = addr
        self.elem_per_warp = (self.addr.rows.stop - self.addr.rows.start) // self.addr.rows.step + 1

        # Fill if default_value is not None
        if default_value is not None:
            self.fill(default_value)

    def __del__(self):
        """
        Frees the tensor memory.
        """
        memory.free(self.addr)

    def fill(self, value):
        """
        Writes the given value to all elements of the Tensor
        """
        func = {Type.int32: driver.writeMulti_int, Type.float32: driver.writeMulti_float, Type.bool: driver.writeMulti_bool}[self.dtype]
        func(self.addr.warps, self.addr.rows, self.addr.index, value)

    def __getaddr(self, x):
        """
        Computes the address for the given element index
        """

        if x < 0 or x >= self.nelem:
            raise RuntimeError("Index out of range.")

        warp = self.addr.warps.start + (x // self.elem_per_warp) * self.addr.warps.step
        row = self.addr.rows.start + (x % self.elem_per_warp) * self.addr.rows.step
        index = self.addr.index
        return (warp, row, index)

    def __getitem__(self, x):
        """
        Returns the element at the given index or a view for a given slice
        """

        # Returns a view of the Tensor
        if isinstance(x, slice):

            # Default values for the slice
            start = 0 if x.start is None else x.start
            start = self.nelem + start if start < 0 else start
            step = 1 if x.step is None else x.step
            stop = self.nelem if x.stop is None else x.stop
            stop = self.nelem + stop if stop < 0 else stop
            # Modify stop to be inclusive
            stop = start + ((stop - 1 - start) // step) * step
            nelem = (stop - start) // step + 1
            # Create a new slice object for the child Tensor
            my_slicing = self.get_slicing()
            new_slicing = slice(my_slicing.start + start * my_slicing.step,
                                my_slicing.start + stop * my_slicing.step + 1,
                                my_slicing.step * step)

            # Compute the start/stop addresses
            start_addr = self.__getaddr(start)
            stop_addr = self.__getaddr(stop)

            # If the slice stride divides the number of elements per warp
            if self.elem_per_warp % step == 0:

                row_step = step * self.addr.rows.step

                # If the slice is constrained to a single warp
                if start_addr[0] == stop_addr[0]:
                    
                    warps = driver.RangeMask(start_addr[0], start_addr[0], 1)
                    rows = driver.RangeMask(start_addr[1], stop_addr[1], row_step)
                    return TensorView(self, new_slicing, shape=nelem, addr=memory.Address(warps, rows, self.addr.index))
                
                # If the slice is repeated across several warps
                else:

                    if start_addr[1] - row_step >= self.addr.rows.start:
                        raise RuntimeError("Chosen slice not supported due to different warp row masks.")
                    if stop_addr[1] + row_step <= self.addr.rows.stop:
                        raise RuntimeError("Chosen slice not supported due to different warp row masks.")
                    
                    warps = driver.RangeMask(start_addr[0], stop_addr[0], self.addr.warps.step)
                    rows = driver.RangeMask(start_addr[1], stop_addr[1], row_step)
                    return TensorView(self, new_slicing, shape=nelem, addr=memory.Address(warps, rows, self.addr.index))

            elif step % self.elem_per_warp == 0:

                warp_step = (step // self.elem_per_warp) * self.addr.warps.step

                warps = driver.RangeMask(start_addr[0], stop_addr[0], warp_step)
                rows = driver.RangeMask(start_addr[1], start_addr[1], 1)
                return TensorView(self, new_slicing, shape=nelem, addr=memory.Address(warps, rows, self.addr.index))

            else:
                raise RuntimeError("The slice step size must divide the number of elements per warp or vice-versa.")

        # Reads a single element from the Tensor
        else:

            func = {Type.int32: driver.read_int, Type.float32: driver.read_float, Type.bool: driver.read_bool}[self.dtype]
            return func(*self.__getaddr(x))
        
    def __setitem__(self, x, value):
        """
        Updates the element at the given index
        """

        # If tensor[a:b:c] = value, where x = slice(a, b, c)
        if isinstance(x, slice):

            # Create a view of the self tensor according to the slice
            sliced_self = self.__getitem__(x)
            
            # If the value is a Tensor
            if(isinstance(value, Tensor)):
                value.copy(sliced_self)

            # Else assumes the value is a scalar
            else:
                sliced_self.fill(value)

        # Else, if writing to a single element
        else:
            func = {Type.int32: driver.write_int, Type.float32: driver.write_float, Type.bool: driver.write_bool}[self.dtype]
            func(*self.__getaddr(x), value)

    def __unaryOp(self, func, dtype=None):
        """
        Performs the unary arithmetic operation.
        """

        res = self.empty_like(dtype=dtype)
        func(self.addr.index, res.addr.index, self.addr.warps, self.addr.rows)
        return res

    def __binaryOp(self, other, func, dtype=None):
        """
        Performs the binary arithmetic operation.
        """

        if self.dtype != other.dtype:
            raise RuntimeError(f"Binary operation failed: type mismatch.")
        if self.shape != other.shape:
            raise RuntimeError(f"Binary operation failed: shape mismatch.")
        
        # If other Tensor is misaligned
        if (self.addr.warps != other.addr.warps) or (self.addr.rows != other.addr.rows):
            other_copy = self.empty_like()
            other.copy(other_copy)
            other = other_copy
        
        if self.addr.index == other.addr.index:
            other = self.empty_like()
            self.copy(other)

        res = self.empty_like(dtype=None)
        func(self.addr.index, other.addr.index, res.addr.index, self.addr.warps, self.addr.rows)
        return res
    
    def __ternaryOp(self, other1, other2, func, dtype=None):
        """
        Performs the ternary arithmetic operation. Does not perform type-checking.
        """
        
        # If other Tensor is misaligned
        if (self.addr.warps != other1.addr.warps) or (self.addr.rows != other1.addr.rows):
            other1_copy = self.empty_like()
            other1.copy(other1_copy)
            other1 = other1_copy
        if (self.addr.warps != other2.addr.warps) or (self.addr.rows != other2.addr.rows):
            other2_copy = self.empty_like()
            other2.copy(other2_copy)
            other2 = other2_copy
        
        if self.addr.index == other1.addr.index:
            other1 = self.empty_like()
            self.copy(other1)
        if self.addr.index == other2.addr.index:
            other2 = self.empty_like()
            self.copy(other2)
        
        res = self.empty_like(dtype=dtype)
        func(self.addr.index, other1.addr.index, other2.addr.index, res.addr.index, self.addr.warps, self.addr.rows)
        return res

    def __add__(self, other):
        """
        Performs element-parallel addition with another Tensor
        """
        func = {Type.int32: driver.add_int, Type.float32: driver.add_float}[self.dtype]
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return self.__binaryOp(other, func)
    
    def __radd__(self, other):
        """
        Performs element-parallel addition with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return other.__add__(self)
    
    def __neg__(self):
        """
        Performs element-parallel negation
        """
        func = {Type.int32: driver.negate_int, Type.float32: driver.negate_float}[self.dtype]
        return self.__unaryOp(func)
    
    def abs(self):
        """
        Performs element-parallel absolute value
        """
        func = {Type.int32: driver.absolute_int, Type.float32: driver.absolute_float}[self.dtype]
        return self.__unaryOp(func)

    def __sub__(self, other):
        """
        Performs element-parallel subtraction with another Tensor
        """
        func = {Type.int32: driver.subtract_int, Type.float32: driver.subtract_float}[self.dtype]
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return self.__binaryOp(other, func)
    
    def __rsub__(self, other):
        """
        Performs element-parallel subtraction with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return other.__sub__(self)

    def __mul__(self, other):
        """
        Performs element-parallel multiplication with another Tensor
        """
        func = {Type.int32: driver.multiply_int, Type.float32: driver.multiply_float}[self.dtype]
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return self.__binaryOp(other, func)

    def __rmul__(self, other):
        """
        Performs element-parallel multiplication with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return other.__mul__(self)
    
    def __truediv__(self, other):
        """
        Performs element-parallel division with another Tensor
        """
        func = {Type.int32: driver.divide_int, Type.float32: driver.divide_float}[self.dtype]
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return self.__binaryOp(other, func)
    
    def __rtruediv__(self, other):
        """
        Performs element-parallel division with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return other.__truediv__(self)
    
    def __mod__(self, other):
        """
        Performs element-parallel modulo division with another Tensor
        """
        if not (isinstance(other, int) or other.dtype == Type.int32):
            raise RuntimeError("Cannot perform modulo operation on non-int data type.")
        func = {Type.int32: driver.modulo_int}[self.dtype]
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return self.__binaryOp(other, func)
    
    def __rmod__(self, other):
        """
        Performs element-parallel modulo division with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return other.__mod__(self)
    
    def sign(self):
        """
        Returns the sign of the tensor as a binary Tensor
        """
        func = {Type.int32: driver.sign_int, Type.float32: driver.sign_float}[self.dtype]
        return self.__unaryOp(func, dtype=Type.bool)
    
    def zero(self):
        """
        Returns whether each element is zero as a binary Tensor
        """
        func = {Type.int32: driver.zero_int, Type.float32: driver.zero_float}[self.dtype]
        return self.__unaryOp(func, dtype=Type.bool)
    
    def __invert__(self):
        """
        Returns the bitwise NOT of the Tensor
        """
        return self.__unaryOp(driver.bitwiseNot)
    
    def __and__(self, other):
        """
        Returns the bitwise AND of the Tensor with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return self.__binaryOp(other, driver.bitwiseAnd)
    
    def __rand__(self, other):
        """
        Returns the bitwise AND of the Tensor with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return other.__and__(self)
    
    def __xor__(self, other):
        """
        Returns the bitwise XOR of the Tensor with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return self.__binaryOp(other, driver.bitwiseXor)
    
    def __rxor__(self, other):
        """
        Returns the bitwise XOR of the Tensor with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return other.__xor__(self)

    def __or__(self, other):
        """
        Returns the bitwise OR of the Tensor with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return self.__binaryOp(other, driver.bitwiseOr)
    
    def __ror__(self, other):
        """
        Returns the bitwise OR of the Tensor with another Tensor
        """
        other = other if isinstance(other, Tensor) else self.empty_like(default_value=other)
        return other.__or__(self)
    
    def __lt__(self, other):
        """
        Performs element-parallel comparison (less than) with another Tensor
        """
        diff = self - other
        return diff.sign()
    
    def __le__(self, other):
        """
        Performs element-parallel comparison (less than or equal to) with another Tensor
        """
        diff = self - other
        return diff.sign() | diff.zero()
    
    def __gt__(self, other):
        """
        Performs element-parallel comparison (greater than) with another Tensor
        """
        diff = other - self
        return diff.sign()
    
    def __ge__(self, other):
        """
        Performs element-parallel comparison (greater than or equal to) with another Tensor
        """
        diff = other - self
        return diff.sign() | diff.zero()
    
    def __eq__(self, other):
        """
        Performs element-parallel comparison (equal to) with another Tensor
        """
        return (self - other).zero()
    
    def __ne__(self, other):
        """
        Performs element-parallel comparison (not equal to) with another Tensor
        """
        return ~(self - other).zero()
    
    def mux(self, a, b):    

        if self.dtype != Type.bool:
            raise RuntimeError(f"Binary operation failed: type mismatch.")
        if a.dtype != b.dtype:
            raise RuntimeError(f"Binary operation failed: type mismatch.")
        if self.shape != a.shape:
            raise RuntimeError(f"Binary operation failed: shape mismatch.")
        if self.shape != b.shape:
            raise RuntimeError(f"Binary operation failed: shape mismatch.")

        return self.__ternaryOp(a, b, driver.bitwiseMux, dtype=a.dtype)
    
    def copy(self, other):
        """
        Copies the contents of the current Tensor into the given tensor
        """

        if self.dtype != other.dtype:
            raise RuntimeError(f"Copy operation failed: type mismatch.")
        if self.shape != other.shape:
            raise RuntimeError(f"Copy operation failed: shape mismatch.")
                
        # Misaligned addresses, use move operation
        if (self.addr.warps != other.addr.warps) or (self.addr.rows != other.addr.rows):

            for i in range(self.elem_per_warp):

                src_addr = self.__getaddr(i)
                dst_addr = other.__getaddr(i)
                driver.move(src_addr[1], src_addr[2], dst_addr[1], dst_addr[2], dst_addr[0] - src_addr[0], self.addr.warps)

        # Aligned addresses, can use horizontal copy
        else:
            driver.copy(self.addr.index, other.addr.index, self.addr.warps, self.addr.rows)

    def clone(self):
        """
        Creates a Tensor with identical contents to the given Tensor.
        """
        return self.__unaryOp(driver.copy)
    
    def empty_like(self, default_value = None, dtype=None):
        """
        Returns a Tensor like the current Tensor (shape, slicing, dtype, addr) that is unitialized (or initialized with default_value if specified)
        """
        dtype = self.dtype if dtype is None else dtype
        return Tensor(*self.get_parent_tensor().shape, dtype=dtype, ref_tensor=self.get_parent_tensor(), default_value=default_value)[self.get_slicing()]
    
    def zeros_like(self, dtype=None):
        """
        Returns a Tensor like the current Tensor (shape, slicing, dtype, addr) initiated with zeros
        """
        return self.empty_like(default_value=0, dtype=dtype)
    
    def ones_like(self, dtype=None):
        """
        Returns a Tensor like the current Tensor (shape, slicing, dtype, addr) initiated with ones
        """
        return self.empty_like(default_value=1, dtype=dtype)
    
    def reduce(self, func):
        """
        Performs the logarithmic reduction of the given tensor using the specified function (function name as string)
        """
        y = self.clone()
        for i in range(math.ceil(math.log(self.nelem, 2))):
            y[::2 ** (i + 1)] = getattr(y[::2 ** (i + 1)], func)(y[2 ** i::2 ** (i + 1)])
        return y[0]
    
    def sum(self):
        """
        Performs a logarithmic reduction sum
        """
        return self.reduce('__add__')
    
    def prod(self):
        """
        Performs a logarithmic reduction product
        """
        return self.reduce('__mul__')
    
    def sort(self, group_size = None):
        """
        Performs a bitonic sorting operation on the given Tensor. If group_size is specified, then the sorting is performed in independent groups (batched sorting).
        """

        if group_size is None:
            group_size = self.nelem

        res, temp, dir = self.clone(), self.empty_like(), self.empty_like(dtype=Type.bool)

        k = 2
        while k <= group_size:
                
            dir.fill(0)
            for i in range(min(k, dir.nelem - k)):
                dir[k + i::2 * k] = -1

            j = k // 2
            while j > 0:

                for i in range(j):
                    temp[i:-j:2 * j] = res[i + j::2 * j]
                res, temp = compare_and_swap(res, temp, dir)
                for i in range(j):
                    res[i + j::2 * j] = temp[i:-j:2 * j]
                    
                j //= 2
            k *= 2

        return res
    
    def get_parent_tensor(self):
        return self
    
    def get_slicing(self):
        return slice(0, self.nelem, 1)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}): [" + ','.join([str(self[x]) for x in range(self.nelem)]) + ']'

class TensorView(Tensor):
    """
    Represents a non-owning view of a Tensor
    """

    def __init__(self, owner: Tensor, slicing: slice, shape, addr):
        super().__init__(shape, dtype=owner.dtype, addr=addr)
        self.owner = owner
        self.slicing = slicing

    def __del__(self):
        pass

    def get_parent_tensor(self):
        return self.owner.get_parent_tensor()
    
    def get_slicing(self):
        return self.slicing

    def __repr__(self):
        return f"TensorView(shape={self.shape}, dtype={self.dtype}, slicing={self.slicing}): [" + ','.join([str(self[x]) for x in range(self.nelem)]) + ']'

def empty(*args, **kwargs):
    """
    Creates a Tensor with the given parameters.
    """
    return Tensor(*args, **kwargs)

def zeros(*args, **kwargs):
    """
    Creates a Tensor of zeros with the given parameters.
    """
    return Tensor(default_value=0, *args, **kwargs)

def ones(*args, **kwargs):
    """
    Creates a Tensor of ones with the given parameters.
    """
    return Tensor(default_value=1, *args, **kwargs)

def from_numpy(arr: np.ndarray, *args, **kwargs):
    """
    Creates a Tensor given a numpy array object
    """

    # Perform dtype conversion
    if arr.dtype not in FROM_NUMPY_DTYPE_CONVERSION:
        raise NotImplementedError(f"Data type {arr.dtype} not supported.")
    dtype = FROM_NUMPY_DTYPE_CONVERSION[arr.dtype]

    # Allocate empty array
    res = empty(arr.size, dtype=dtype, *args, **kwargs)

    # Copy elements
    for i in range(arr.size):
        res[i] = arr[i]

    return res

def to_numpy(tensor: Tensor):
    """
    Converts the given Tensor to a numpy array
    """

    # Perform dtype conversion
    dtype = TO_NUMPY_DTYPE_CONVERSION[tensor.dtype]

    # Allocate empty array
    res = np.empty(tensor.shape, dtype=dtype)

    # Copy elements
    for i in range(tensor.nelem):
        res[i] = tensor[i]

    return res

def linspace(*args, **kwargs):
    """
    Returns a Tensor initated with linearly-spaced values. See NumPy notation.
    """
    kwargs['dtype'] = TO_NUMPY_DTYPE_CONVERSION[kwargs['dtype'] if 'dtype' in kwargs.keys() else Type.float32]
    return from_numpy(np.linspace(*args, **kwargs))

def arange(*args, **kwargs):
    """
    Returns a Tensor initated with a range of values. See NumPy notation.
    """
    kwargs['dtype'] = TO_NUMPY_DTYPE_CONVERSION[kwargs['dtype'] if 'dtype' in kwargs.keys() else Type.int32]
    return from_numpy(np.arange(*args, **kwargs))

def compare_and_swap(a: Tensor, b: Tensor, dir: Tensor):
    """
    Performs a compare and swap (CAS) operation on the given pair of Tensors
    """
    sel = ((b < a) ^ dir)
    return (~sel).mux(a, b), sel.mux(a, b)
