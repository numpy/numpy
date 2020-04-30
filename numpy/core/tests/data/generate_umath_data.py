import numpy as np
import argparse
import mpmath

parser = argparse.ArgumentParser(
    description='Generate umath data. '
                'Input file can only contain hex numbers in separate rows, '
                'empty lines and lines starting with "#" (comments).')
parser.add_argument('-i', '--input_file', type=str, required=True,
                    help="Name of file with inputs used to generate test data.")
parser.add_argument('-f', '--funcname', required=True,
                    help="Name of mpmath function used to generate data.")
parser.add_argument('-o', '--output_file', type=str, required=True,
                    help="Name of output file. Should be of the form "
                         "umath-validation-set-<funcname>.")
parser.add_argument('--dtype', type=str, required=True,
                    help="dtype of data")
parser.add_argument('--ulperror', type=int, default=2)

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
dtype = getattr(np, args.dtype[3:])
with open(args.input_file, 'r') as input_file:
    with open(args.output_file, 'w') as output_file:
        for i, line in enumerate(lines_to_process(input_file, output_file)):
            arg = from_hex(line, dtype)
            exp = dtype(mpfunc(arg))
            name = args.funcname+str(i+1).zfill(3)
            output_file.write(
                "{} {} {} {} {}  # {} -> {}\n".format(
                    name, args.dtype, args.ulperror, to_hex(arg),
                    to_hex(exp), arg, exp))
