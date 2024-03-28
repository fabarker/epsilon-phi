from setuptools import setup, Extension
import numpy

# Define the extension module
module = Extension('kse', sources=['gkse.cpp'],
                   include_dirs=[numpy.get_include()])

# Run the setup
setup(
    name='kse',
    version='0.1',
    description='Python interface for C++ find_1st function',
    ext_modules=[module])