"""
Setup script for pyhqp - Python bindings for HQP solver
"""

import os

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Get the directory of this setup.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "pyhqp",
        sources=[
            os.path.join(current_dir, "pybind_hqp.cpp"),
        ],
        include_dirs=[
            os.path.join(parent_dir, "include"),
            pybind11.get_include(),
            "/usr/include/eigen3",  # Add Eigen include path
        ],
        language="c++",
        cxx_std=20,
        define_macros=[
            ("VERSION_INFO", '"dev"'),
        ],
    ),
]

setup(
    name="pyhqp",
    version="0.1.1",
    author="Gianluca Garofalo",
    author_email="gianluca.garofalo@outlook.com",
    description="Python bindings for HQP (Hierarchical Quadratic Programming) solver",
    long_description=open(
        os.path.join(parent_dir, "README.md"), encoding="utf-8"
    ).read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pybind11",
    ],
    extras_require={
        "dev": [
            "pytest",
            "numpy",
        ],
    },
)
