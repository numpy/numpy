import numpy as np
from numpy.testing.print_coercion_tables import (
    print_cancast_table,
    print_new_cast_table,
)


def test_print_new_cast_table(capsys):
    # print_cancast_table first: it registers casts with an error-code level.
    print_cancast_table(np.typecodes['All'])
    print_new_cast_table(can_cast=True, legacy=True, flags=True)
    assert capsys.readouterr().out
