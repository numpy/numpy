"""
Test the scalar constructors, which also do type-coercion
"""
import fractions
import platform
import types
from typing import Any, Type
from math import copysign

import pytest
import numpy as np

from numpy.testing import assert_equal, assert_raises, IS_MUSL


class TestAsIntegerRatio:
    # derived in part from the cpython test "test_floatasratio"

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    @pytest.mark.parametrize("f, ratio", [
        (0.875, (7, 8)),
        (-0.875, (-7, 8)),
        (0.0, (0, 1)),
        (11.5, (23, 2)),
        ])
    def test_small(self, ftype, f, ratio):
        assert_equal(ftype(f).as_integer_ratio(), ratio)

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    def test_simple_fractions(self, ftype):
        R = fractions.Fraction
        assert_equal(R(0, 1),
                     R(*ftype(0.0).as_integer_ratio()))
        assert_equal(R(5, 2),
                     R(*ftype(2.5).as_integer_ratio()))
        assert_equal(R(1, 2),
                     R(*ftype(0.5).as_integer_ratio()))
        assert_equal(R(-2100, 1),
                     R(*ftype(-2100.0).as_integer_ratio()))

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    def test_errors(self, ftype):
        assert_raises(OverflowError, ftype('inf').as_integer_ratio)
        assert_raises(OverflowError, ftype('-inf').as_integer_ratio)
        assert_raises(ValueError, ftype('nan').as_integer_ratio)

    def test_against_known_values(self):
        R = fractions.Fraction
        assert_equal(R(1075, 512),
                     R(*np.half(2.1).as_integer_ratio()))
        assert_equal(R(-1075, 512),
                     R(*np.half(-2.1).as_integer_ratio()))
        assert_equal(R(4404019, 2097152),
                     R(*np.single(2.1).as_integer_ratio()))
        assert_equal(R(-4404019, 2097152),
                     R(*np.single(-2.1).as_integer_ratio()))
        assert_equal(R(4728779608739021, 2251799813685248),
                     R(*np.double(2.1).as_integer_ratio()))
        assert_equal(R(-4728779608739021, 2251799813685248),
                     R(*np.double(-2.1).as_integer_ratio()))
        # longdouble is platform dependent

    @pytest.mark.parametrize("ftype, frac_vals, exp_vals", [
        # dtype test cases generated using hypothesis
        # first five generated cases per dtype
        (np.half, [0.0, 0.01154830649280303, 0.31082276347447274,
                   0.527350517124794, 0.8308562335072596],
                  [0, 1, 0, -8, 12]),
        (np.single, [0.0, 0.09248576989263226, 0.8160498218131407,
                     0.17389442853722373, 0.7956044195067877],
                    [0, 12, 10, 17, -26]),
        (np.double, [0.0, 0.031066908499895136, 0.5214135908877832,
                     0.45780736035689296, 0.5906586745934036],
                    [0, -801, 51, 194, -653]),
        pytest.param(
            np.longdouble,
            [0.0, 0.20492557202724854, 0.4277180662199366, 0.9888085019891495,
             0.9620175814461964],
            [0, -7400, 14266, -7822, -8721],
            marks=[
                pytest.mark.skipif(
                    np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double"),
                pytest.mark.skipif(
                    platform.machine().startswith("ppc"),
                    reason="IBM double double"),
            ]
        )
    ])
    def test_roundtrip(self, ftype, frac_vals, exp_vals):
        for frac, exp in zip(frac_vals, exp_vals):
            f = np.ldexp(ftype(frac), exp)
            assert f.dtype == ftype
            n, d = f.as_integer_ratio()

            try:
                nf = np.longdouble(n)
                df = np.longdouble(d)
                if not np.isfinite(df):
                    raise OverflowError
            except (OverflowError, RuntimeWarning):
                # the values may not fit in any float type
                pytest.skip("longdouble too small on this platform")

            assert_equal(nf / df, f, "{}/{}".format(n, d))


class TestIsInteger:
    @pytest.mark.parametrize("str_value", ["inf", "nan"])
    @pytest.mark.parametrize("code", np.typecodes["Float"])
    def test_special(self, code: str, str_value: str) -> None:
        cls = np.dtype(code).type
        value = cls(str_value)
        assert not value.is_integer()

    @pytest.mark.parametrize(
        "code", np.typecodes["Float"] + np.typecodes["AllInteger"]
    )
    def test_true(self, code: str) -> None:
        float_array = np.arange(-5, 5).astype(code)
        for value in float_array:
            assert value.is_integer()

    @pytest.mark.parametrize("code", np.typecodes["Float"])
    def test_false(self, code: str) -> None:
        float_array = np.arange(-5, 5).astype(code)
        float_array *= 1.1
        for value in float_array:
            if value == 0:
                continue
            assert not value.is_integer()


class TestClassGetItem:
    @pytest.mark.parametrize("cls", [
        np.number,
        np.integer,
        np.inexact,
        np.unsignedinteger,
        np.signedinteger,
        np.floating,
    ])
    def test_abc(self, cls: Type[np.number]) -> None:
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    def test_abc_complexfloating(self) -> None:
        alias = np.complexfloating[Any, Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.complexfloating

    @pytest.mark.parametrize("arg_len", range(4))
    def test_abc_complexfloating_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            assert np.complexfloating[arg_tup]
        else:
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            with pytest.raises(TypeError, match=match):
                np.complexfloating[arg_tup]

    @pytest.mark.parametrize("cls", [np.generic, np.flexible, np.character])
    def test_abc_non_numeric(self, cls: Type[np.generic]) -> None:
        with pytest.raises(TypeError):
            cls[Any]

    @pytest.mark.parametrize("code", np.typecodes["All"])
    def test_concrete(self, code: str) -> None:
        cls = np.dtype(code).type
        with pytest.raises(TypeError):
            cls[Any]

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.number[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.number[arg_tup]

    def test_subscript_scalar(self) -> None:
        assert np.number[Any]


class TestBitCount:
    # derived in part from the cpython test "test_bit_count"

    @pytest.mark.parametrize("itype", np.sctypes['int']+np.sctypes['uint'])
    def test_small(self, itype):
        for a in range(max(np.iinfo(itype).min, 0), 128):
            msg = f"Smoke test for {itype}({a}).bit_count()"
            assert itype(a).bit_count() == bin(a).count("1"), msg

    def test_bit_count(self):
        for exp in [10, 17, 63]:
            a = 2**exp
            assert np.uint64(a).bit_count() == 1
            assert np.uint64(a - 1).bit_count() == exp
            assert np.uint64(a ^ 63).bit_count() == 7
            assert np.uint64((a - 1) ^ 510).bit_count() == exp - 8

class TestFloatHex:
    # derived in part from the cpython test "HexFloatTestCase"

    def identical(self, x, y):
        # check that floats x and y are identical, or that both
        # are NaNs
        if np.isnan(x) or np.isnan(y):
            if np.isnan(x) == np.isnan(y):
                return True
        elif x == y and (
                x != 0.0 or np.copysign(1.0, x) == np.copysign(1.0, y)
            ):
            return True

        return False

    def roundtrip(self, x):
        return x.fromhex(x.hex())

    @pytest.mark.parametrize("ftype", np.sctypes['float'])
    def test_ends(self, ftype):
        finfo = np.finfo(ftype)

        # 2**MINEXP
        assert self.identical(
                finfo.smallest_normal, np.ldexp(ftype(1.0), finfo.minexp))

        # 2**(MINEXP - NMANT)
        assert self.identical(
                finfo.smallest_subnormal,
                np.ldexp(ftype(1.0), finfo.minexp-finfo.nmant))

        # 2**(-NMANT)
        assert self.identical(
                finfo.eps, np.ldexp(ftype(1.0), -finfo.nmant))

        # 2**(MAXEXP-1) - 2**((MAXEXP - 1) - NMANT - 1)
        assert self.identical(
                finfo.max,
                2.*(np.ldexp(ftype(1.0),
                    finfo.maxexp-1) - np.ldexp(ftype(1.0),
                        finfo.maxexp - finfo.nmant - 2)
                    )
                )

    @pytest.mark.parametrize("ftype", np.sctypes['float'])
    def test_roundtrip(self, ftype):
        finfo = np.finfo(ftype)
        fltMax, fltMin = finfo.max, finfo.min
        fltEps, fltTiny = finfo.eps, finfo.tiny

        for x in [fltMax, fltMin, fltEps, fltTiny, 0.0, np.nan, np.inf]:
            assert self.identical(x, self.roundtrip(x))

    @pytest.mark.parametrize("ftype", np.sctypes['float'])
    def test_from_hex(self, ftype):
        finfo = np.finfo(ftype)
        MINEXP = finfo.minexp
        MAXEXP = finfo.maxexp
        NMANT = finfo.nmant
        TINY = finfo.smallest_subnormal

        # two spellings of infinity, with optional signs; case-insensitive
        assert self.identical(ftype.fromhex('inf'), np.inf)
        assert self.identical(ftype.fromhex('+Inf'), np.inf)
        assert self.identical(ftype.fromhex('-INF'), -np.inf)
        assert self.identical(ftype.fromhex('iNf'), np.inf)
        assert self.identical(ftype.fromhex('Infinity'), np.inf)
        assert self.identical(ftype.fromhex('+INFINITY'), np.inf)
        assert self.identical(ftype.fromhex('-infinity'), -np.inf)
        assert self.identical(ftype.fromhex('-iNFiNitY'), -np.inf)

        # nans with optional sign; case insensitive
        assert self.identical(ftype.fromhex('nan'), np.nan)
        assert self.identical(ftype.fromhex('+NaN'), np.nan)
        assert self.identical(ftype.fromhex('-NaN'), np.nan)
        assert self.identical(ftype.fromhex('-nAN'), np.nan)

        # variations in input format
        assert self.identical(ftype.fromhex('1'), 1.0)
        assert self.identical(ftype.fromhex('+1'), 1.0)
        assert self.identical(ftype.fromhex('1.'), 1.0)
        assert self.identical(ftype.fromhex('1.0'), 1.0)
        assert self.identical(ftype.fromhex('1.0p0'), 1.0)
        assert self.identical(ftype.fromhex('01'), 1.0)
        assert self.identical(ftype.fromhex('01.'), 1.0)
        assert self.identical(ftype.fromhex('0x1'), 1.0)
        assert self.identical(ftype.fromhex('0x1.'), 1.0)
        assert self.identical(ftype.fromhex('0x1.0'), 1.0)
        assert self.identical(ftype.fromhex('+0x1.0'), 1.0)
        assert self.identical(ftype.fromhex('0x1p0'), 1.0)
        assert self.identical(ftype.fromhex('0X1p0'), 1.0)
        assert self.identical(ftype.fromhex('0X1P0'), 1.0)
        assert self.identical(ftype.fromhex('0x1P0'), 1.0)
        assert self.identical(ftype.fromhex('0x1.p0'), 1.0)
        assert self.identical(ftype.fromhex('0x1.0p0'), 1.0)
        assert self.identical(ftype.fromhex('+0x1p0'), 1.0)
        assert self.identical(ftype.fromhex('0x01p0'), 1.0)
        assert self.identical(ftype.fromhex('0x1p00'), 1.0)
        assert self.identical(ftype.fromhex(' 0x1p0 '), 1.0)
        assert self.identical(ftype.fromhex('\n 0x1p0'), 1.0)
        assert self.identical(ftype.fromhex('0x1p0 \t'), 1.0)
        assert self.identical(ftype.fromhex('0xap0'), 10.0)
        assert self.identical(ftype.fromhex('0xAp0'), 10.0)
        assert self.identical(ftype.fromhex('0xaP0'), 10.0)
        assert self.identical(ftype.fromhex('0xAP0'), 10.0)
        assert self.identical(ftype.fromhex('0xbep0'), 190.0)
        assert self.identical(ftype.fromhex('0xBep0'), 190.0)
        assert self.identical(ftype.fromhex('0xbEp0'), 190.0)
        assert self.identical(ftype.fromhex('0XBE0P-4'), 190.0)
        assert self.identical(ftype.fromhex('0xBEp0'), 190.0)
        assert self.identical(ftype.fromhex('0xB.Ep4'), 190.0)
        assert self.identical(ftype.fromhex('0x.BEp8'), 190.0)
        assert self.identical(ftype.fromhex('0x.0BEp12'), 190.0)

        # moving the point around
        pi = ftype.fromhex('0x1.921fb54442d18p1')
        assert self.identical(ftype.fromhex('0x.006487ed5110b46p11'), pi)
        assert self.identical(ftype.fromhex('0x.00c90fdaa22168cp10'), pi)
        assert self.identical(ftype.fromhex('0x.01921fb54442d18p9'), pi)
        assert self.identical(ftype.fromhex('0x.03243f6a8885a3p8'), pi)
        assert self.identical(ftype.fromhex('0x.06487ed5110b46p7'), pi)
        assert self.identical(ftype.fromhex('0x.0c90fdaa22168cp6'), pi)
        assert self.identical(ftype.fromhex('0x.1921fb54442d18p5'), pi)
        assert self.identical(ftype.fromhex('0x.3243f6a8885a3p4'), pi)
        assert self.identical(ftype.fromhex('0x.6487ed5110b46p3'), pi)
        assert self.identical(ftype.fromhex('0x.c90fdaa22168cp2'), pi)
        assert self.identical(ftype.fromhex('0x1.921fb54442d18p1'), pi)
        assert self.identical(ftype.fromhex('0x3.243f6a8885a3p0'), pi)
        assert self.identical(ftype.fromhex('0x6.487ed5110b46p-1'), pi)
        assert self.identical(ftype.fromhex('0xc.90fdaa22168cp-2'), pi)
        assert self.identical(ftype.fromhex('0x19.21fb54442d18p-3'), pi)
        assert self.identical(ftype.fromhex('0x32.43f6a8885a3p-4'), pi)
        assert self.identical(ftype.fromhex('0x64.87ed5110b46p-5'), pi)
        assert self.identical(ftype.fromhex('0xc9.0fdaa22168cp-6'), pi)
        assert self.identical(ftype.fromhex('0x192.1fb54442d18p-7'), pi)
        assert self.identical(ftype.fromhex('0x324.3f6a8885a3p-8'), pi)
        assert self.identical(ftype.fromhex('0x648.7ed5110b46p-9'), pi)
        assert self.identical(ftype.fromhex('0xc90.fdaa22168cp-10'), pi)
        assert self.identical(ftype.fromhex('0x1921.fb54442d18p-11'), pi)
        # ... TODO Make it go towards NMANT
        assert self.identical(ftype.fromhex('0x1921fb54442d1.8p-47'), pi)
        assert self.identical(ftype.fromhex('0x3243f6a8885a3p-48'), pi)
        assert self.identical(ftype.fromhex('0x6487ed5110b46p-49'), pi)
        assert self.identical(ftype.fromhex('0xc90fdaa22168cp-50'), pi)
        assert self.identical(ftype.fromhex('0x1921fb54442d18p-51'), pi)
        assert self.identical(ftype.fromhex('0x3243f6a8885a30p-52'), pi)
        assert self.identical(ftype.fromhex('0x6487ed5110b460p-53'), pi)
        assert self.identical(ftype.fromhex('0xc90fdaa22168c0p-54'), pi)
        assert self.identical(ftype.fromhex('0x1921fb54442d180p-55'), pi)

        # results that should overflow...
        large_values = [
            f"-0x1p{MAXEXP}", f"0x1p+{MAXEXP + 1}", f"+0X1p{MAXEXP + 5}",
            f"-0x1p+{MAXEXP + 50}", f"0X1p123456789123456789",
            f"+0X.8p+{MAXEXP + 1}", f"+0x0.8p{MAXEXP + 1}",
            f"-0x0.4p{MAXEXP + 2}", f"0X2p+{MAXEXP - 1}",
            f"0x2.p{MAXEXP - 1}", f"-0x2.0p+{MAXEXP - 1}",
            f"+0X4p+{MAXEXP - 2}",
            f"0x1.{'f' * (NMANT // 4 + 1)}p+{MAXEXP - 1}",
            # for below case, last `f` is just to be sure
            # for 128. It can also be `9` for 64, etc
            f"-0X1.{'f' * (NMANT // 4)}fp{MAXEXP - 1}",
            f"+0x3.{'f' * (NMANT // 4 + 1)}p{MAXEXP - 2}",
            # XXX How are we getting to 970 for 64?
            # f"0x3fffffffffffffp+{MAXEXP - NMANT}",
            f"0x10000000000000000p{MAXEXP - NMANT}",
            f"-0Xffffffffffffffffp{MAXEXP - NMANT}",
        ]

        for x in large_values:
            with pytest.raises(OverflowError):
                result = ftype.fromhex(x)

        # ...and those that round to +-max float
        self.identical(ftype.fromhex(
        f"+0x1.{'f' * (NMANT // 4)}p+{MAXEXP - 1}"),
                finfo.max)
        self.identical(ftype.fromhex(
        f"-0x1.{'f' * (NMANT // 4)}7p+{MAXEXP - 1}"),
                -finfo.max)
        self.identical(ftype.fromhex(
        f"+0x1.{'f' * (NMANT // 4)}7{'f' * (NMANT // 4 )}p+{MAXEXP - 1}"),
                finfo.max)

        # zeros
        assert self.identical(ftype.fromhex(
            f'0x0p0'), 0.0)
        assert self.identical(ftype.fromhex(
            f'0x0p{MAXEXP // 100 * 100}'), 0.0)
        assert self.identical(ftype.fromhex(
            f'-0x0p{MAXEXP - 1}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'0X0p{MAXEXP}'), 0.0)
        assert self.identical(ftype.fromhex(
            f'-0x0p{MAXEXP + 1}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'0X0p{MAXEXP // 100 * 100 * 2}'), 0.0)
        assert self.identical(ftype.fromhex(
            f'0x0p123456789123456789'), 0.0)
        assert self.identical(ftype.fromhex(
            f'-0X0p-0'), -0.0)
        assert self.identical(ftype.fromhex(
            f'-0X0p-{MAXEXP // 100 * 100}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'0x0p-{MAXEXP - 1}'), 0.0)
        assert self.identical(ftype.fromhex(
            f'-0X0p-{MAXEXP}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'-0x0p-{MAXEXP + 1}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'-0x0p-{MAXEXP + NMANT - 4}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'0X0p-{MAXEXP + NMANT - 3}'), 0.0)
        assert self.identical(ftype.fromhex(
            f'-0x0p-{MAXEXP + NMANT - 2}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'0x0p-{MAXEXP + NMANT - 1}'), 0.0)
        assert self.identical(ftype.fromhex(
            f'0X0p-{MAXEXP + NMANT}'), 0.0)
        assert self.identical(ftype.fromhex(
            f'-0X0p-{MAXEXP // 100 * 100 * 2}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'-0x0p-123456789123456789'), -0.0)

        # values that should underflow to 0
        assert self.identical(ftype.fromhex(
            f'0X1p-{MAXEXP + NMANT - 1}'), 0.0)
        assert self.identical(ftype.fromhex(
            f'-0X1p-{MAXEXP + NMANT - 1}'), -0.0)
        assert self.identical(ftype.fromhex(
            f'-0x1p-123456789123456789'), -0.0)
        assert self.identical(
            ftype.fromhex(
                f'0x1.{"0" * (NMANT // 4 + 1)}1p-{MAXEXP + NMANT - 1}'), TINY)
        assert self.identical(
            ftype.fromhex(f'-0x1.1p-{MAXEXP + NMANT - 1}'), -TINY)
        assert self.identical(
            ftype.fromhex(
                f'0x1.{"f" * (NMANT // 4 + 1)}p-{MAXEXP + NMANT - 1}'), TINY)

        # check round-half-even is working correctly near 0 ...

        # ... and near MIN ...

        # ... and near 1.0.

        # Regression test for a corner-case bug reported in b.p.o. 44954
        # refer to python/cpython for more details

    @pytest.mark.parametrize("ftype", np.sctypes['float'])
    def test_invalid_inputs(self, ftype):
        invalid_inputs = [
            'infi',  # misspelt infinities and nans
            '-Infinit',
            '++inf',
            '-+Inf',
            '--nan',
            '+-NaN',
            'snan',
            'NaNs',
            'nna',
            'an',
            'nf',
            'nfinity',
            'inity',
            'iinity',
            '0xnan',
            '',
            ' ',
            'x1.0p0',
            '0xX1.0p0',
            '+ 0x1.0p0',  # internal whitespace
            '- 0x1.0p0',
            '0 x1.0p0',
            '0x 1.0p0',
            '0x1 2.0p0',
            '+0x1 .0p0',
            '0x1. 0p0',
            '-0x1.0 1p0',
            '-0x1.0 p0',
            '+0x1.0p +0',
            '0x1.0p -0',
            '0x1.0p 0',
            '+0x1.0p+ 0',
            '-0x1.0p- 0',
            '++0x1.0p-0',  # double signs
            '--0x1.0p0',
            '+-0x1.0p+0',
            '-+0x1.0p0',
            '0x1.0p++0',
            '+0x1.0p+-0',
            '-0x1.0p-+0',
            '0x1.0p--0',
            '0x1.0.p0',
            '0x.p0',  # no hex digits before or after point
            '0x1,p0',  # wrong decimal point character
            '0x1pa',
            '0x1p\uff10',  # fullwidth Unicode digits
            '\uff10x1p0',
            '0x\uff11p0',
            '0x1.\uff10p0',
            '0x1p0 \n 0x2p0',
            '0x1p0\0 0x1p0',  # embedded null byte is not end of string
            ]
        for x in invalid_inputs:
            with pytest.raises(ValueError):
                result = ftype.fromhex(x)

    @pytest.mark.parametrize("ftype", np.sctypes['float'])
    @pytest.mark.parametrize("value_pairs", [
            ('inf', np.inf),
            ('-Infinity', -np.inf),
            ('nan', np.nan),
            ('1.0', 1.0),
            ('-0x.2', -0.125),
            ('-0.0', -0.0)
        ])
    @pytest.mark.parametrize("lead", [
            '', ' ', '\t', '\n', '\n \t', '\f', '\v', '\r'
        ])
    @pytest.mark.parametrize("trail", [
            '', ' ', '\t', '\n', '\n \t', '\f', '\v', '\r'
        ])
    def test_whitespace(self, ftype, value_pairs, lead, trail):
        inp, expected = value_pairs
        got = ftype.fromhex(lead + inp + trail)
        assert self.identical(got, expected)
