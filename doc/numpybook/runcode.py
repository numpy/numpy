"""
This script takes a lyx file and runs the python code in it.
 Then rewrites the lyx file again.

Each section of code portion is assumed to be in the same namespace
where a from numpy import * has been applied

 If a PYNEW inside a Note is encountered, the name space is restarted

The output (if any) is replaced in the file
 by the output produced during the code run.

Options:
  -n name of code section  (default MyCode)

"""
from __future__ import division, absolute_import, print_function

import sys
import optparse
import io
import re
import os

newre = re.compile(r"\\begin_inset Note.*PYNEW\s+\\end_inset", re.DOTALL)

def getoutput(tstr, dic):
    print("\n\nRunning...")
    print(tstr, end=' ')
    tempstr = io.StringIO()
    sys.stdout = tempstr
    code = compile(tstr, '<input>', 'exec')
    try:
        res = eval(tstr, dic)
        sys.stdout = sys.__stdout__
    except SyntaxError:
        try:
            res = None
            exec(code, dic)
        finally:
            sys.stdout = sys.__stdout__
    if res is None:
        res = tempstr.getvalue()
    else:
        res = tempstr.getvalue() + '\n' + repr(res)
    if res != '':
        print("\nOutput is")
        print(res, end=' ')
    return res

# now find the code in the code segment
def getnewcodestr(substr, dic):
    end = substr.find('\\layout ')
    lines = substr[:end].split('\\newline')
    outlines = []
    first = 1
    cmd = ''
    lines.append('dummy')
    for line in lines:
        line = line.strip()
        if (line[:3]=='>>>') or (line == 'dummy'):
            # we have a new output
            pyoutstr = getoutput(cmd, dic).strip()
            if pyoutstr != '':
                pyout = pyoutstr.split('\n')
                outlines.extend(pyout)
            cmd = line[4:]
        elif (line[:3]=='...'):
            # continuation output
            cmd += "\n%s" % line[4:]
        else:
            # first line or output
            if first:
                first = 0
                cmd = line
            else:
                continue
        if line != 'dummy':
            outlines.append(line)
    return "\n\\newline \n".join(outlines), end


def runpycode(lyxstr, name='MyCode'):
    schobj = re.compile(r"\\layout %s\s+>>> " % name)
    outstr = io.StringIO()
    num = 0
    indx = []
    for it in schobj.finditer(lyxstr):
        indx.extend([it.start(), it.end()])
        num += 1

    if num == 0:
        print("Nothing found for %s" % name)
        return lyxstr

    start = 0
    del indx[0]
    indx.append(len(lyxstr))
    edic = {}
    exec('from numpy import *', edic)
    exec('set_printoptions(linewidth=65)', edic)
    # indx now contains [st0,en0, ..., stN,enN]
    #  where stX is the start of code segment X
    #  and enX is the start of \layout MyCode for
    #  the X+1 code section (or string length if X=N)
    for k in range(num):
        # first write everything up to the start of the code segment
        substr = lyxstr[start:indx[2*k]]
        outstr.write(substr)
        if start > 0:
            mat = newre.search(substr)
            # if PYNEW found, then start a new namespace
            if mat:
                edic = {}
                exec('from numpy import *', edic)
                exec('set_printoptions(linewidth=65)', edic)
        # now find the code in the code segment
        # endoutput will contain the index just past any output
        #  already present in the lyx string.
        substr = lyxstr[indx[2*k]:indx[2*k+1]]
        lyxcodestr, endcode = getnewcodestr(substr, edic)
        # write the lyx for the input + new output
        outstr.write(lyxcodestr)
        outstr.write('\n')
        start = endcode + indx[2*k]

    outstr.write(lyxstr[start:])
    return outstr.getvalue()


def main(args):
    usage = "%prog {options} filename"
    parser = optparse.OptionParser(usage)
    parser.add_option('-n','--name', default='MyCode')

    options, args = parser.parse_args(args)
    if len(args) < 1:
        parser.error("incorrect number of arguments")

    os.system('cp -f %s %s.bak' % (args[0], args[0]))
    fid = file(args[0])
    str = fid.read()
    fid.close()
    print("Processing %s" % options.name)
    newstr = runpycode(str, options.name)
    fid = file(args[0],'w')
    fid.write(newstr)
    fid.close()

if __name__ == "__main__":
    main(sys.argv[1:])
