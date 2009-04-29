"""Simple script to compute the api hash of the current API as defined by
numpy_api_order and ufunc_api_order."""
from os.path import join, dirname

from genapi import fullapi_hash

if __name__ == '__main__':
    curdir = dirname(__file__)
    files = [join(curdir, 'numpy_api_order.txt'),
             join(curdir, 'ufunc_api_order.txt')]
    print fullapi_hash(files)
