# setup.py

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "solutions",
    ext_modules = cythonize('solutions.pyx', compiler_directives={"language_level": "3"}), # accepts a glob pattern
)
