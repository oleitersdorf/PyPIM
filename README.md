# PyPIM: Integrating Digital Processing-in-Memory from Microarchitectural Design to Python Tensors
## Overview
This repository includes the framework proposed in the following paper,

`Orian Leitersdorf, Ronny Ronen, Shahar Kvatinsky, “PyPIM: Integrating Digital Processing-in-Memory from Microarchitectural Design to Python Tensors,” Accepted to IEEE/ACM MICRO 2024.`

The framework enables high-level programming of PIM applications with significant ease. The framework benefits from the high
flexibility of tensor-based Python (e.g., NumPy, PyTorch, TensorFlow) to provide the user with simple operations that can already be executed today (through the simulator backend):
```
>>> import pypim as pim
>>> x = pim.Tensor(8, dtype=pim.float32)
>>> x
Tensor(shape=(8,), dtype=float32): [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
>>> 
>>> x[2] = 2.5
>>> x[3] = 1.25
>>> x[4] = 2.25
>>> x
Tensor(shape=(8,), dtype=float32): [0.0,0.0,2.5,1.25,2.25,0.0,0.0,0.0]
>>> 
>>> x[::2]
TensorView(shape=(4,), dtype=float32, slicing=slice(0, 7, 2)): [0.0,2.5,2.25,0.0]
>>> x[::2].sum()
4.75
>>> x[::2].sort()
TensorView(shape=(4,), dtype=float32, slicing=slice(0, 7, 2)): [0.0,0.0,2.25,2.5]
```
This also enables the user to assemble new PIM routines from existing arithmetic functions, such as:
```
import pypim as pim
def myFunc(a: pim.Tensor, b: pim.Tensor):    
    # Parallel multiplication and addition 
    return a * (1 + b)
```

The repository is split into four parts: (1) the underlying GPU-accelerated simulator, (2) the microarchitectural driver, (3) the development library,
and (4) a series of test scripts that serve as the benchmarks of PyPIM.

## User Information
### Dependencies
The simulation environment is implemented via `CUDA` to enable fast execution of many samples in parallel. Therefore,
the project requires the following dependencies:
1. CUDA 12.0 (with a capable GPU of at least 8GB DRAM)
2. Python installation (tested with 3.10)
2. Compiler for C++ 17 (or higher)

### Installation

The development library may be installed using `pip` from the project directory as follows:
```
pip install -e .
```

The installation can then be verified by running `python main.py`, with the expected result being `32.0`.

### Organization
The repository is organized into the following directories:
- `csrc`: this directory contains the C++ source code for the simulator and the driver.
- `pypim`: this directory contains the Python source code for the development library.
- `tests`: this directory contains the source code for the tests.
- `results`: this directory contains the raw results of the tests.
- `main.py`: this file contains the example script from the paper.

# Instruction-Set-Architecture (ISA)

The full instruction-set-architecture (ISA) supported in CUDA-PIM is as follows:

| Operation                  | Int Support | Float Support |
|----------------------------|:-----------:|:-------------:|
| **Arithmetic**             |             |               |
| Addition                   | &#10003;    | &#10003;      |
| Subtraction                | &#10003;    | &#10003;      |
| Multiplication             | &#10003;    | &#10003;      |
| Division                   | &#10003;    | &#10003;      |
| Modulo                     | &#10003;    |               |
| Negation                   | &#10003;    | &#10003;      |
| **Comparison**             |             |               |
| Less than (or equal to)    | &#10003;    | &#10003;      |
| Greater than (or equal to) | &#10003;    | &#10003;      |
| Equal                      | &#10003;    | &#10003;      |
| **Bitwise**                |             |               |
| Bitwise Not                | &#10003;    | &#10003;      |
| Bitwise And                | &#10003;    | &#10003;      |
| Bitwise Or                 | &#10003;    | &#10003;      |
| Bitwise Xor                | &#10003;    | &#10003;      |
| **Miscellaneous**          |             |               |
| Sign                       | &#10003;    | &#10003;      |
| Zero                       | &#10003;    | &#10003;      |
| Abs                        | &#10003;    | &#10003;      |
| Mux                        | &#10003;    | &#10003;      |