from __future__ import division, absolute_import, print_function

import sys
import copy
import binascii

LONG_DOUBLE_REPRESENTATION_SRC = r"""
/* "before" is 16 bytes to ensure there's no padding between it and "x".
 *    We're not expecting any "long double" bigger than 16 bytes or with
 *       alignment requirements stricter than 16 bytes.  */
typedef %(type)s test_type;

struct {
        char         before[16];
        test_type    x;
        char         after[8];
} foo = {
        { '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
          '\001', '\043', '\105', '\147', '\211', '\253', '\315', '\357' },
        -123456789.0,
        { '\376', '\334', '\272', '\230', '\166', '\124', '\062', '\020' }
};
"""

def pyod(filename):
    """Python implementation of the od UNIX utility (od -b, more exactly).

    Parameters
    ----------
    filename : str
        name of the file to get the dump from.

    Returns
    -------
    out : seq
        list of lines of od output

    Note
    ----
    We only implement enough to get the necessary information for long double
    representation, this is not intended as a compatible replacement for od.
    """
    def _pyod2():
        out = []

        fid = open(filename, 'r')
        try:
            yo = [int(oct(int(binascii.b2a_hex(o), 16))) for o in fid.read()]
            for i in range(0, len(yo), 16):
                line = ['%07d' % int(oct(i))]
                line.extend(['%03d' % c for c in yo[i:i+16]])
                out.append(" ".join(line))
            return out
        finally:
            fid.close()

    def _pyod3():
        out = []

        fid = open(filename, 'rb')
        try:
            yo2 = [oct(o)[2:] for o in fid.read()]
            for i in range(0, len(yo2), 16):
                line = ['%07d' % int(oct(i)[2:])]
                line.extend(['%03d' % int(c) for c in yo2[i:i+16]])
                out.append(" ".join(line))
            return out
        finally:
            fid.close()

    if sys.version_info[0] < 3:
        return _pyod2()
    else:
        return _pyod3()

_BEFORE_SEQ = ['000', '000', '000', '000', '000', '000', '000', '000',
              '001', '043', '105', '147', '211', '253', '315', '357']
_AFTER_SEQ = ['376', '334', '272', '230', '166', '124', '062', '020']

_IEEE_DOUBLE_BE = ['301', '235', '157', '064', '124', '000', '000', '000']
_IEEE_DOUBLE_LE = _IEEE_DOUBLE_BE[::-1]
_INTEL_EXTENDED_12B = ['000', '000', '000', '000', '240', '242', '171', '353',
                       '031', '300', '000', '000']
_INTEL_EXTENDED_16B = ['000', '000', '000', '000', '240', '242', '171', '353',
                       '031', '300', '000', '000', '000', '000', '000', '000']
_IEEE_QUAD_PREC_BE = ['300', '031', '326', '363', '105', '100', '000', '000',
                      '000', '000', '000', '000', '000', '000', '000', '000']
_IEEE_QUAD_PREC_LE = _IEEE_QUAD_PREC_BE[::-1]
_DOUBLE_DOUBLE_BE = ['301', '235', '157', '064', '124', '000', '000', '000'] + \
                    ['000'] * 8

def long_double_representation(lines):
    """Given a binary dump as given by GNU od -b, look for long double
    representation."""

    # Read contains a list of 32 items, each item is a byte (in octal
    # representation, as a string). We 'slide' over the output until read is of
    # the form before_seq + content + after_sequence, where content is the long double
    # representation:
    #  - content is 12 bytes: 80 bits Intel representation
    #  - content is 16 bytes: 80 bits Intel representation (64 bits) or quad precision
    #  - content is 8 bytes: same as double (not implemented yet)
    read = [''] * 32
    saw = None
    for line in lines:
        # we skip the first word, as od -b output an index at the beginning of
        # each line
        for w in line.split()[1:]:
            read.pop(0)
            read.append(w)

            # If the end of read is equal to the after_sequence, read contains
            # the long double
            if read[-8:] == _AFTER_SEQ:
                saw = copy.copy(read)
                if read[:12] == _BEFORE_SEQ[4:]:
                    if read[12:-8] == _INTEL_EXTENDED_12B:
                        return 'INTEL_EXTENDED_12_BYTES_LE'
                elif read[:8] == _BEFORE_SEQ[8:]:
                    if read[8:-8] == _INTEL_EXTENDED_16B:
                        return 'INTEL_EXTENDED_16_BYTES_LE'
                    elif read[8:-8] == _IEEE_QUAD_PREC_BE:
                        return 'IEEE_QUAD_BE'
                    elif read[8:-8] == _IEEE_QUAD_PREC_LE:
                        return 'IEEE_QUAD_LE'
                    elif read[8:-8] == _DOUBLE_DOUBLE_BE:
                        return 'DOUBLE_DOUBLE_BE'
                elif read[:16] == _BEFORE_SEQ:
                    if read[16:-8] == _IEEE_DOUBLE_LE:
                        return 'IEEE_DOUBLE_LE'
                    elif read[16:-8] == _IEEE_DOUBLE_BE:
                        return 'IEEE_DOUBLE_BE'

    if saw is not None:
        raise ValueError("Unrecognized format (%s)" % saw)
    else:
        # We never detected the after_sequence
        raise ValueError("Could not lock sequences (%s)" % saw)
