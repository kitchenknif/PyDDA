from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("ott_funcs.pyx, misc.pyx, dda_si_integration_funcs.pyx")
)