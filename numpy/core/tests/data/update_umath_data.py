import numpy as np
import argparse
import mpmath

parser = argparse.ArgumentParser(
    description="Update umath data. "
                "Input file can only contain lines of format described "
                "in README possibly without `name` entry. Script can change "
                "`name` entries to guarantee their uniqueness.")
parser.add_argument('-i', '--input_file', type=str, required=True,
                    help="Name of data file.")
parser.add_argument('-f', '--funcname', required=True,
                    help="Name of mpmath function used to generate data.")
parser.add_argument('-o', '--output_file', type=str, required=True,
                    help="Name of output file. Should be of the form "
                         "umath-validation-set-<funcname>")

args = parser.parse_args()


def from_hex(s, dtype):
    barr = bytearray.fromhex(s[2:])
    barr.reverse()
    return np.frombuffer(barr, dtype=dtype)[0]


def to_hex(x):
    barr = bytearray(x)
    barr.reverse()
    return "0x" + barr.hex()


def lines_to_process(input_file, output_file):
    for line in input_file:
        if not line.strip() or line[0] == '#':
            output_file.write(line)
            continue
        if '#' in line:
            line = line[:line.index('#')]

        yield line.strip()


mpfunc = getattr(mpmath, args.funcname)
tested_func = getattr(np, args.funcname)
with open(args.input_file, 'r') as input_file:
    with open(args.output_file, 'w') as output_file:
        for i, line in enumerate(lines_to_process(input_file, output_file)):
            vals = line.split(' ')
            if len(vals) == 5:
                _, dtype_str, ulperror, arg_hex, exp_hex = vals
            else:
                dtype_str, ulperror, arg_hex, exp_hex = vals

            dtype = getattr(np, dtype_str[3:])
            name = args.funcname+str(i+1).zfill(3)
            arg = from_hex(arg_hex, dtype)
            exp = from_hex(exp_hex, dtype)
            output_file.write(
                "{} {} {} {} {}  # {} -> {}\n".format(
                    name, dtype_str, ulperror, arg_hex,
                    exp_hex, arg, exp))

            res = mpfunc(arg)
            if isinstance(res, mpmath.mpc):
                res = res.real
            mpmath_exp = dtype(res)
            if exp != mpmath_exp:
                print(f"WARNING: test {name} in {args.output_file}:")
                print(f"x input from file:         {arg_hex} ({arg})")
                print(f"expected exp(x) from file: {exp_hex} ({exp})")
                print(f"mpmath result (real part): {to_hex(mpmath_exp)} ({mpmath_exp})\n")
