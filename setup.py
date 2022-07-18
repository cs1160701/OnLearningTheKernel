#!/usr/bin/env python

# This code is adapted from this source 
# https://github.com/idiap/fast-transformers

"""Setup fast transformers"""

from functools import lru_cache
from itertools import dropwhile
from os import path
from setuptools import find_packages, setup
from subprocess import DEVNULL, call

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError as e:
    raise ImportError(
        ("PyTorch is required to install pytorch-fast-transformers. Please "
         "install your favorite version of PyTorch, we support 1.3.1, 1.5.0 "
         "and >=1.6"),
        name=e.name,
        path=e.path
    ) from e


@lru_cache(None)
def cuda_toolkit_available():
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("fast_transformers", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def get_extensions():
    extensions = [
        CppExtension(
            "fast_transformers.hashing.hash_cpu",
            sources=[
                "fast_transformers/hashing/hash_cpu.cpp"
            ],
            extra_compile_args=["-fopenmp", "-ffast-math"]
        ),
        CppExtension(
            "fast_transformers.aggregate.aggregate_cpu",
            sources=[
               "fast_transformers/aggregate/aggregate_cpu.cpp"
            ],
            extra_compile_args=["-fopenmp", "-ffast-math"]
        ),
        CppExtension(
            "fast_transformers.clustering.hamming.cluster_cpu",
            sources=[
               "fast_transformers/clustering/hamming/cluster_cpu.cpp"
            ],
            extra_compile_args=["-fopenmp", "-ffast-math"]
        ),
        CppExtension(
            "fast_transformers.sparse_product.sparse_product_cpu",
            sources=[
                "fast_transformers/sparse_product/sparse_product_cpu.cpp"
            ],
            extra_compile_args=["-fopenmp", "-ffast-math"]
        ),
        CppExtension(
            "fast_transformers.sparse_product.clustered_sparse_product_cpu",
            sources=[
                "fast_transformers/sparse_product/clustered_sparse_product_cpu.cpp"
            ],
            extra_compile_args=["-fopenmp", "-ffast-math"]
        ),
        CppExtension(
            "fast_transformers.causal_product.causal_product_cpu",
            sources=[
                "fast_transformers/causal_product/causal_product_cpu.cpp"
            ],
            extra_compile_args=["-fopenmp", "-ffast-math"]
        ),
        CppExtension(
            "fast_transformers.local_product.local_product_cpu",
            sources=[
                "fast_transformers/local_product/local_product_cpu.cpp"
            ],
            extra_compile_args=["-fopenmp", "-ffast-math"]
        )
    ]
    if cuda_toolkit_available():
        from torch.utils.cpp_extension import CUDAExtension
        extensions += [
            CUDAExtension(
                "fast_transformers.hashing.hash_cuda",
                sources=[
                    "fast_transformers/hashing/hash_cuda.cu",
                ],
                extra_compile_args=["-arch=compute_50"]
            ),
            CUDAExtension(
                "fast_transformers.aggregate.aggregate_cuda",
                sources=[
                    "fast_transformers/aggregate/aggregate_cuda.cu"
                ],
                extra_compile_args=["-arch=compute_50"]
            ),
            CUDAExtension(
                "fast_transformers.aggregate.clustered_aggregate_cuda",
                sources=[
                    "fast_transformers/aggregate/clustered_aggregate_cuda.cu"
                ],
                extra_compile_args=["-arch=compute_50"]
            ),
            CUDAExtension(
                "fast_transformers.clustering.hamming.cluster_cuda",
                sources=[
                    "fast_transformers/clustering/hamming/cluster_cuda.cu"
                ],
                extra_compile_args=["-arch=compute_50"]
            ),
            CUDAExtension(
                "fast_transformers.sparse_product.sparse_product_cuda",
                sources=[
                    "fast_transformers/sparse_product/sparse_product_cuda.cu"
                ],
                extra_compile_args=["-arch=compute_50"]
            ),
            CUDAExtension(
                "fast_transformers.sparse_product.clustered_sparse_product_cuda",
                sources=[
                    "fast_transformers/sparse_product/clustered_sparse_product_cuda.cu"
                ],
                extra_compile_args=["-arch=compute_50"]
            ),
            CUDAExtension(
                "fast_transformers.causal_product.causal_product_cuda",
                sources=[
                    "fast_transformers/causal_product/causal_product_cuda.cu"
                ],
                extra_compile_args=["-arch=compute_50"]
            ),
            CUDAExtension(
                "fast_transformers.local_product.local_product_cuda",
                sources=[
                    "fast_transformers/local_product/local_product_cuda.cu"
                ],
                extra_compile_args=["-arch=compute_50"]
            )
        ]
    return extensions


def setup_package():
    with open("README.rst") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="pytorch-fast-transformers",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        long_description_content_type="text/x-rst",
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
        ],
        packages=find_packages(exclude=["docs", "tests", "scripts", "examples"]),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
        install_requires=["torch"]
    )


if __name__ == "__main__":
    setup_package()