"""Simple script to compute the api hash of the current API.

The API has is defined by numpy_api_order and ufunc_api_order.

"""
from __future__ import division, absolute_import, print_function

from os.path import dirname

from genapi import fullapi_hash
import numpy_api


if __name__ == '__main__':
    curdir = dirname(__file__)
    print(fullapi_hash(numpy_api.full_api))
