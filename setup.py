import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def find_cuda():
    """Find CUDA installation path."""
    cuda_paths = [
        "/usr/local/cuda",
        "/usr",
    ]
    for path in cuda_paths:
        nvcc = os.path.join(path, "bin", "nvcc")
        if os.path.exists(nvcc):
            return path
        if path == "/usr" and os.path.exists("/usr/bin/nvcc"):
            return "/usr"
    raise RuntimeError("CUDA not found")


CUDA_HOME = find_cuda()


class CUDAExtension(Extension):
    """Marker class for extensions that need CUDA compilation."""
    pass


class BuildExtension(build_ext):
    """Custom build_ext that handles .cu files with nvcc."""

    def build_extensions(self):
        for ext in self.extensions:
            self._build_extension(ext)

    def _build_extension(self, ext):
        # Separate .cu and .cpp files
        cu_sources = [s for s in ext.sources if s.endswith('.cu')]
        cpp_sources = [s for s in ext.sources if s.endswith('.cpp')]

        # Create build directory
        os.makedirs(self.build_temp, exist_ok=True)

        objects = []

        # Compile .cu files with nvcc
        for cu_file in cu_sources:
            obj_file = os.path.join(
                self.build_temp, os.path.basename(cu_file) + ".o"
            )
            self._compile_cuda(cu_file, obj_file)
            objects.append(obj_file)

        # Compile .cpp files with g++
        for cpp_file in cpp_sources:
            obj_file = os.path.join(
                self.build_temp, os.path.basename(cpp_file) + ".o"
            )
            self._compile_cpp(cpp_file, obj_file)
            objects.append(obj_file)

        # Link everything into .so
        self._link(objects, ext)

    def _compile_cuda(self, source, output):
        """Compile a .cu file with nvcc."""
        import pybind11

        cmd = [
            "nvcc",
            "-c", source,
            "-o", output,
            "-Xcompiler", "-fPIC",
            f"-I{CUDA_HOME}/include",
            f"-I{pybind11.get_include()}",
            "-std=c++17",
        ]
        print(f"Compiling CUDA: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    def _compile_cpp(self, source, output):
        """Compile a .cpp file with g++."""
        import pybind11
        from sysconfig import get_paths

        python_include = get_paths()["include"]

        cmd = [
            "g++",
            "-c", source,
            "-o", output,
            "-fPIC",
            f"-I{CUDA_HOME}/include",
            f"-I{pybind11.get_include()}",
            f"-I{python_include}",
            "-Itensor/csrc",
            "-std=c++17",
        ]
        print(f"Compiling C++: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    def _link(self, objects, ext):
        """Link object files into a shared library."""
        import sysconfig

        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        output = os.path.join("tensor", "_C" + ext_suffix)

        # Determine CUDA library path
        if CUDA_HOME == "/usr":
            lib_dir = "/usr/lib/x86_64-linux-gnu"
        else:
            lib_dir = f"{CUDA_HOME}/lib64"

        cmd = [
            "g++",
            "-shared",
            *objects,
            "-o", output,
            f"-L{lib_dir}",
            "-lcudart",
        ]
        print(f"Linking: {' '.join(cmd)}")
        subprocess.check_call(cmd)


ext_modules = [
    CUDAExtension(
        "tensor._C",
        sources=[
            "tensor/csrc/bindings.cpp",
            "tensor/csrc/tensor.cu",
        ],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
