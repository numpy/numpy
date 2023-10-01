""" Distributor init file

Distributors: you can add a _distributor_init_local.py file to support
particular distributions of numpy.

For example, this is a good place to put any BLAS/LAPACK initialization code.
"""

try:
    import _distributor_init_local
except ModuleNotFoundError:
    pass
