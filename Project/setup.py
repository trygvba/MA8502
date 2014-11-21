from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Airfoil-test Cytonized',
  ext_modules = cythonize("airfoiltest_total_cy.pyx"),
)
