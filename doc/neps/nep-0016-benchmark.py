import perf
import abc
import numpy as np

class NotArray:
    pass

class AttrArray:
    __array_implementer__ = True

class ArrayBase(abc.ABC):
    pass

class ABCArray1(ArrayBase):
    pass

class ABCArray2:
    pass


ArrayBase.register(ABCArray2)

not_array = NotArray()
attr_array = AttrArray()
abc_array_1 = ABCArray1()
abc_array_2 = ABCArray2()

# Make sure ABC cache is primed
isinstance(not_array, ArrayBase)
isinstance(abc_array_1, ArrayBase)
isinstance(abc_array_2, ArrayBase)

runner = perf.Runner()
def t(name, statement):
    runner.timeit(name, statement, globals=globals())


t("np.asarray([])", "np.asarray([])")
arrobj = np.array([])
t("np.asarray(arrobj)", "np.asarray(arrobj)")

t("attr, False",
  "getattr(not_array, '__array_implementer__', False)")
t("attr, True",
  "getattr(attr_array, '__array_implementer__', False)")

t("ABC, False", "isinstance(not_array, ArrayBase)")
t("ABC, True, via inheritance", "isinstance(abc_array_1, ArrayBase)")
t("ABC, True, via register", "isinstance(abc_array_2, ArrayBase)")
