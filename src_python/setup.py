from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="wrapper",
        sources=["wrapper.pyx", "./src_c/min_sum.c"],  # incluir wrapper y tu código C
        include_dirs=["./include"],  # para encontrar mylib.h
    ),
    Extension(
        name="wrapper_csc",
        sources=["wrapper_csc.pyx", "./src_c/min_sum_csc.c"],  # incluir wrapper y tu código C
        include_dirs=["./include"],  # para encontrar mylib.h
    )
]

setup(
    name="wrapper_csc",
    ext_modules=cythonize(extensions),
)