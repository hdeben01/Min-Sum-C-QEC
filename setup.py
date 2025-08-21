from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="wrapper",
        sources=["wrapper.pyx", "min_sum.c"],  # incluir wrapper y tu código C
        include_dirs=["./include"],  # para encontrar mylib.h
    )
]

setup(
    name="wrapper",
    ext_modules=cythonize(extensions),
)