# Copyright (c) 2023, NumPy Developers.

from typing import TYPE_CHECKING

from .module import Module

if TYPE_CHECKING:
    from ...interpreter import Interpreter

def initialize(interpreter: 'Interpreter') -> Module:
    return Module()
