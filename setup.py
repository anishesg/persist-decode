from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

sources = [
    os.path.join(csrc_dir, "bindings.cpp"),
    os.path.join(src_dir, "persistent_layer.cu"),
    os.path.join(src_dir, "reference.cu"),
    os.path.join(src_dir, "persistent_multi_layer.cu"),
]

nvcc_flags = [
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-use_fast_math",
    "-std=c++17",
]

setup(
    name="persist_decode",
    version="0.1.0",
    description="Persistent-kernel transformer decode with shared-memory-resident hidden state",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="persist_decode._C",
            sources=sources,
            include_dirs=[src_dir, csrc_dir],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
