from setuptools import setup, Extension
import numpy

# Define the extension module
cFunc_module = Extension('cDates', sources=['cDates.cpp'],
                   include_dirs=[numpy.get_include()])

# Setup function
setup(
    name='cDates',
    version='1.0',
    description='A Python module that does date things with C extension.',
    ext_modules=[cFunc_module]
)