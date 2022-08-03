
# NOTE: Please avoid the use of numpy.testing since NPYV intrinsics
# may be involved in their functionality.
import sys
import ctypes
from dataclasses import dataclass
from enum import IntEnum, auto
import operator

class _ContainerID(IntEnum):
    SCALAR = 1
    ARRAY = auto()
    VEC = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    MASK = auto()

class _ElementID(IntEnum):
    VOID = 0
    BOOL = auto()
    UINT8 = auto()
    INT8 = auto()
    UINT16 = auto()
    INT16 = auto()
    UINT32 = auto()
    INT32 = auto()
    UINT64 = auto()
    INT64 = auto()
    FLOAT = auto()
    DOUBLE = auto()

class _Sequence:
    def __init__(self, *vals):
        data_type = self.element_ctype * len(vals)
        cvt = float if self.element_id > _ElementID.INT64 else int
        data = data_type(*[cvt(v) for v in vals])
        self._bytearray = bytearray(
            bytes([self.container_id, self.element_id]) +
            bytes(data)
        )

    def data(self):
        data_type = self.element_ctype * len(self)
        return data_type.from_buffer(self._bytearray, 2)

    def __getitem__(self, index):
        data = self.data()
        try:
            data = data[index]
        except IndexError:
            raise IndexError(type(self).__name__ + " index out of range")
        if isinstance(index, int):
            tp = _Type.byid(_ContainerID.SCALAR, self.element_id)
            return tp(data)
        return type(self)(*data)

    def __setitem__(self, index, value):
        data = self.data()
        data[index] = value

    def __len__(self):
        return (len(self._bytearray) - 2) // self.element_size

    def __iter__(self):
        yield from self._iterate_items()

    def _iterate_items(self):
        tp = _Type.byid(_ContainerID.SCALAR, self.element_id)
        data = self.data()
        for d in data:
            yield tp(d)

    def __str__(self):
        return str(tuple(self.data()))

    def __repr__(self):
        return type(self).__name__ + str(self)

    def to_list(self):
        return [v for v in self.data()]

    def to_bytearray(self):
        return self._bytearray

    @classmethod
    def from_bytes(cls, raw):
        self = super().__new__(cls)
        self._bytearray = bytearray(
            bytes([cls.container_id, cls.element_id]) + bytes(raw)
        )
        return self

class _SequenceX2(_Sequence):
    def __init__(self, val0, val1):
        if len(val0) != len(val1):
            raise ValueError("Mismatched sequence length")
        super().__init__(*val0, *val1)

    def __len__(self):
        return 2

    def data(self, index):
        vlen = super().__len__() // len(self)
        vstart = index * vlen * self.element_size
        data_type = self.element_ctype * vlen
        return data_type.from_buffer(self.to_bytearray(), 2 + vstart)

    def __getitem__(self, index):
        if not isinstance(index, int):
            return NotImplemented
        vlen = len(self)
        if index >= vlen:
            raise IndexError(type(self).__name__ + " index out of range")
        tp = _Type.byid(_ContainerID.VEC, self.element_id)
        return tp(*self.data(index))

    def __setitem__(self, index, value):
        if not isinstance(index, int):
            return NotImplemented
        vlen = len(self)
        if index >= vlen:
            raise IndexError(type(self).__name__ + " index out of range")
        data = self.data(index)
        value = tuple(value)
        if len(data) != len(value):
            raise ValueError(
                f"Mismatched sequence length, expected {len(data)},"
                f" got({len(value)})"
            )
        for i, v in enumerate(value):
            data[i] = v

    def __iter__(self):
        yield from self._iterate_items()

    def _iterate_items(self):
        tp = _Type.byid(_ContainerID.VEC, self.element_id)
        for i in range(len(self)):
            yield tp(*self.data(i))

    def __str__(self):
        return str(tuple(
            str(self[i]) for i in range(len(self))
        )).replace("'", '')

    def __repr__(self):
        return type(self).__name__ + str(self)

class _SequenceX3(_SequenceX2):
    def __init__(self, val0, val1, val2):
        if not (len(val0) == len(val1) and len(val0) == len(val2)):
            raise ValueError("Mismatched sequence length")
        _Sequence.__init__(self, *val0, *val1, *val2)

    def __len__(self):
        return 3

class _SequenceX4(_SequenceX2):
    def __init__(self, val0, val1, val2, val3):
        if not (
            len(val0) == len(val1) and
            len(val0) == len(val2) and
            len(val0) == len(val3)
        ):
            raise ValueError("Mismatched sequence length")
        _Sequence.__init__(self, *val0, *val1, *val2, *val3)

    def __len__(self):
        return 4

class _Mask(_Sequence):
    def __init__(self, *vals):
        m = (1 << (self.element_size*8)) - 1
        data_type = self.element_ctype * len(vals)
        data = data_type(*[m if v else 0 for v in vals])
        self._bytearray = bytearray(
            bytes([self.container_id, self.element_id]) +
            bytes(data)
        )

class _Scalar:
    def __init__(self, val):
        cvt = float if self.element_id > _ElementID.INT64 else int
        data = self.element_ctype(cvt(val))
        self._bytearray = bytearray(
            bytes([self.container_id, self.element_id]) +
            bytes(data)
        )

    def data(self):
        return self.element_ctype.from_buffer(self._bytearray, 2)

    def __str__(self):
        return str(self.data().value)

    def __repr__(self):
        return type(self).__name__ + f"({str(self)})"

    def __int__(self):
        return int(self.data().value)

    def __float__(self):
        return float(self.data().value)

    def __bool__(self):
        return bool(self.data().value)

    def to_bytearray(self):
        return self._bytearray

    @classmethod
    def from_bytes(cls, raw):
        self = super().__new__(cls)
        self._bytearray = bytearray(
            bytes([cls.container_id, cls.element_id]) + bytes(raw)
        )
        return self

class _Type(type):
    _insta_cache = {}
    _prop_by_eid = {
        _ElementID.VOID: ("void", None),
        _ElementID.BOOL: ("bool_", ctypes.c_bool),
        _ElementID.UINT8: ("uint8_t", ctypes.c_uint8),
        _ElementID.INT8: ("int8_t", ctypes.c_int8),
        _ElementID.UINT16: ("uint16_t", ctypes.c_uint16),
        _ElementID.INT16: ("int16_t", ctypes.c_int16),
        _ElementID.UINT32: ("uint32_t", ctypes.c_uint32),
        _ElementID.INT32: ("int32_t", ctypes.c_int32),
        _ElementID.UINT64: ("uint64_t", ctypes.c_uint64),
        _ElementID.INT64: ("int64_t", ctypes.c_int64),
        _ElementID.FLOAT: ("float_", ctypes.c_float),
        _ElementID.DOUBLE: ("double", ctypes.c_double),
    }

    _prop_by_cid = {
        _ContainerID.SCALAR: ('{}', _Scalar),
        _ContainerID.ARRAY: ('Array({})', _Sequence),
        _ContainerID.VEC: ('Vec({})', _Sequence),
        _ContainerID.VEC2: ('Vec2({})', _SequenceX2),
        _ContainerID.VEC3: ('Vec3({})', _SequenceX3),
        _ContainerID.VEC4: ('Vec4({})', _SequenceX4),
        _ContainerID.MASK: ('Mask({})', _Mask),
    }

    def __new__(cls, dummy, bases, attrs):
        container_id = attrs['container_id']
        element_id = attrs['element_id']
        cache_key = (container_id, element_id)
        instance = cls._insta_cache.get(cache_key)
        if instance:
            return instance
        element_name, element_ctype = cls._prop_by_eid[element_id]
        container_name, container_base = cls._prop_by_cid[container_id]
        bases += (container_base,)
        name = container_name.format(element_name)
        attrs.update(dict(
            element_ctype = element_ctype,
            element_size = ctypes.sizeof(element_ctype)
        ))
        cls._gen_cmp(container_id, element_id, attrs)
        cls._gen_bitwise(container_id, element_id, attrs)
        cls._gen_arithm(container_id, element_id, attrs)
        instance = super().__new__(cls, name, bases, attrs)
        _Type._insta_cache[cache_key] = instance
        return instance

    @classmethod
    def byid(cls, container_id, element_id):
        return _Type(cls, (), dict(
            container_id = container_id,
            element_id = element_id
        ))

    @staticmethod
    def _gen_cmp(container_id, element_id, attrs):
        if container_id == _ContainerID.SCALAR:
            if element_id > _ElementID.INT64:
                cvt = float
            else:
                cvt = int
        else:
            cvt = tuple

        def gen_op(self, other, cvt, op, els):
            try:
                rval = cvt(other)
            except TypeError as e:
                return els
            return op(cvt(self), rval)

        for opstr, els in (
            ('eq', False),
            ('ne', False),
            ('lt', NotImplemented),
            ('le', NotImplemented),
            ('gt', NotImplemented),
            ('ge', NotImplemented)
        ):
            op = getattr(operator, opstr)
            attrs[f'__{opstr}__'] = \
                lambda self, other, cvt=cvt, op=op, els=els: gen_op(
                    self, other, cvt, op, els
                )

    @staticmethod
    def _gen_bitwise(container_id, element_id, attrs):
        def gen_binop(self, other, op):
            try:
                a = bytes(self.data())
                b = bytes(other.data())
            except AttributeError:
                return NotImplemented
            if len(a) != len(b):
                return NotImplemented
            return type(self).from_bytes(
                bytes([op(x, y) for x, y in zip(a, b)])
            )

        def gen_unop(self, op):
            try:
                a = bytes(self.data())
            except AttributeError:
                return NotImplemented
            return type(self).from_bytes(bytes([op(x) & 0xFF for x in a]))

        for opstr, op, in (
            ('and', operator.and_),
            ('or', operator.or_),
            ('xor', operator.xor)
        ):
            attrs[f'__{opstr}__'] = \
                lambda self, other, op=op: gen_binop(self, other, op)

        for opstr, op, in (
            ('invert', operator.invert),
        ):
            attrs[f'__{opstr}__'] = \
                lambda self, op=op: gen_unop(self, op)

    @staticmethod
    def _gen_arithm(container_id, element_id, attrs):
        if container_id == _ContainerID.SCALAR:
            if element_id >= _ElementID.FLOAT:
                cvt = float
            else:
                cvt = int
            operations = (
                ('add', 'add', cvt),
                ('sub', 'sub', cvt),
                ('mul', 'mul', cvt),
                ('floordiv', 'floordiv', cvt),
                ('radd', 'add', cvt),
                ('rsub', 'sub', cvt),
                ('rmul', 'mul', cvt),
                ('rfloordiv', 'floordiv', cvt),
            )

            def gen_op(self, other, cvt, op):
                try:
                    lval = cvt(other)
                except TypeError as e:
                    return NotImplemented
                return type(self)(op(cvt(self), lval))
        else:
            operations = (
                ('add', 'add', tuple),
                ('sub', 'sub', tuple),
                ('mul', 'mul', int),
                ('radd', 'add', tuple),
                ('rsub', 'sub', tuple)
            )

            def gen_op(self, other, cvt, op):
                try:
                    lval = cvt(other)
                except TypeError as e:
                    return NotImplemented
                return type(self)(*op(tuple(self), lval))

        for opname, opstr, cvt in operations:
            op = getattr(operator, opstr)
            attrs[f'__{opname}__'] = lambda self, other, cvt=cvt, op=op: \
                gen_op(self, other, cvt, op)

class bool_(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.BOOL

class uint8_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.UINT8

class int8_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.INT8

class uint16_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.UINT16

class int16_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.INT16

class uint32_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.UINT32

class int32_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.INT32

class uint64_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.UINT64

class int64_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.INT64

class size_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = {4: _ElementID.UINT32, 8: _ElementID.UINT64}.get(
                 ctypes.sizeof(ctypes.c_size_t))

class intptr_t(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = {4: _ElementID.INT32, 8: _ElementID.INT64}.get(
                 ctypes.sizeof(ctypes.c_void_p))

class int_(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = {
        2: _ElementID.INT16,
        4: _ElementID.INT32,
        8: _ElementID.INT64
    }.get(ctypes.sizeof(ctypes.c_int))

class float_(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.FLOAT

class double(metaclass=_Type):
    container_id = _ContainerID.SCALAR
    element_id = _ElementID.DOUBLE

class Array(_Type):
    def __new__(cls, element_type):
        s = cls.byid(_ContainerID.ARRAY, element_type.element_id)
        return s

class Vec(_Type):
    def __new__(cls, element_type):
        s = cls.byid(_ContainerID.VEC, element_type.element_id)
        return s

class Mask(_Type):
    def __new__(cls, element_type):
        s = cls.byid(_ContainerID.MASK, element_type.element_id)
        return s

class Vec2(_Type):
    def __new__(cls, element_type):
        return cls.byid(_ContainerID.VEC2, element_type.element_id)

class Vec3(_Type):
    def __new__(cls, element_type):
        return cls.byid(_ContainerID.VEC3, element_type.element_id)

class Vec4(_Type):
    def __new__(cls, element_type):
        return cls.byid(_ContainerID.VEC4, element_type.element_id)


class WrapIntrinsic:
    def __init__(self, name, intrin):
        self.intrin = intrin
        self.name = name
        self.signatures_str = self._gen_signatures_str(
            name, intrin.signatures)

    def __call__(self, *args):
        try:
            ret = self.intrin(*[a.to_bytearray() for a in args])
        except (TypeError, AttributeError) as e:
            targs = ', '.join([type(a).__name__ for a in args])
            raise TypeError(
                f"no matching signature to call {self.name}({targs})\n"
                f"only the following signatures are supported:\n" +
                self.signatures_str
            )
        if isinstance(ret, bytearray):
            tp = _Type.byid(ret[0], ret[1])
            return tp.from_bytes(ret[2:])
        return ret

    @staticmethod
    def _gen_signatures_str(intrin_name, sigs):
        type_name = lambda cid, eid: (
            "void" if eid == _ElementID.VOID else
            _Type.byid(cid, eid).__name__
        )
        ret = []
        for sig in sigs:
            sig = list(zip(sig[0::2], sig[1::2]))
            ret_tname = type_name(sig[0][0], sig[0][1])
            args_tnames = [type_name(cid, eid) for cid, eid in sig[1:]]
            args_tnames = ', '.join(args_tnames)
            ret += [f'{ret_tname} {intrin_name}({args_tnames})']
        return '\n'.join(ret)

class SimdExtention:
    def __init__(self, name, mod_ext):
        self.__name__ = name
        # not supported by the platform/CPU
        if not mod_ext:
            return
        for name, val in mod_ext.__dict__.items():
            if not hasattr(val, "signatures"):
                setattr(self, name, val)
                continue
            setattr(self, name, WrapIntrinsic(name, val))

    def __repr__(self):
        return '_simd.' + self.__name__

    def __str__(self):
        return self.__name__


from ._intrinsics import *

wrap_targets = {}
for name, simd_ext in targets.items():
    wrap_targets[name] = SimdExtention(name, simd_ext)
    globals()[name] = wrap_targets[name]
targets = wrap_targets

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
