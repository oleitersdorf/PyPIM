# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import subprocess

class CustomBuildExt(build_ext):
    def run(self):
        # Compile simulator CUDA shared library
        subprocess.run(["nvcc", "-shared", "-o", 
                        "pypim/libsimulator.so", "csrc/simulator/simulator.cu", 
                        "-lcudart", "-Icsrc", "-Xcompiler", "-fPIC"])
        
        # Run original build_ext run() method
        build_ext.run(self)

setup(
    name='pypim',
    version='0.0.1',
    packages=find_packages(),
    ext_modules=[
        Pybind11Extension(
            "pypim.driver",
            sources=["csrc/driver/driver.cpp"],
            include_dirs=["csrc", "csrc/simulator", "csrc/driver"],
            extra_objects=["pypim/libsimulator.so"],
            extra_compile_args=["-O3"]
        ),
    ],
    cmdclass={"build_ext": CustomBuildExt},
)
