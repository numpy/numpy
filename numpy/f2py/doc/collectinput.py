#!/usr/bin/env python
"""
collectinput - Collects all files that are included to a main Latex document
               with \input or \include commands. These commands must be
               in separate lines.

Copyright 1999 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Pearu Peterson

Usage:
    collectinput <infile> <outfile>
    collectinput <infile>           # <outfile>=inputless_<infile>
    collectinput                    # in and out are stdin and stdout

"""
from __future__ import division, absolute_import, print_function

__version__ = "0.0"

stdoutflag=0
import sys
import fileinput
import re

if sys.version_info[0] >= 3:
    from subprocess import getoutput
else:
    from commands import getoutput

try: fn=sys.argv[2]
except:
    try: fn='inputless_'+sys.argv[1]
    except: stdoutflag=1
try: fi=sys.argv[1]
except: fi=()
if not stdoutflag:
    sys.stdout=open(fn,'w')

nonverb=r'[\w\s\\&=\^\*\.\{\(\)\[\?\+\$/]*(?!\\verb.)'
input=re.compile(nonverb+r'\\(input|include)\*?\s*\{?.*}?')
comment=re.compile(r'[^%]*%')

for l in fileinput.input(fi):
    l=l[:-1]
    l1=''
    if comment.match(l):
        m=comment.match(l)
        l1=l[m.end()-1:]
        l=l[:m.end()-1]
    m=input.match(l)
    if m:
        l=l.strip()
        if l[-1]=='}': l=l[:-1]
        i=m.end()-2
        sys.stderr.write('>>>>>>')
        while i>-1 and (l[i] not in [' ','{']): i=i-1
        if i>-1:
            fn=l[i+1:]
            try: f=open(fn,'r'); flag=1; f.close()
            except:
                try: f=open(fn+'.tex','r'); flag=1;fn=fn+'.tex'; f.close()
                except: flag=0
            if flag==0:
                sys.stderr.write('Could not open a file: '+fn+'\n')
                print(l+l1)
                continue
            elif flag==1:
                sys.stderr.write(fn+'\n')
                print('%%%%% Begin of '+fn)
                print(getoutput(sys.argv[0]+' < '+fn))
                print('%%%%% End of '+fn)
        else:
            sys.stderr.write('Could not extract a file name from: '+l)
            print(l+l1)
    else:
        print(l+l1)
sys.stdout.close()
