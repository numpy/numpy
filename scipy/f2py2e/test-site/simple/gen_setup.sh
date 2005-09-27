#!/bin/sh

f2py=../../f2py2e.py

$f2py -m scalar src/scalar.f --setup  --overwrite-setup
$f2py -m array src/array.f --setup  --overwrite-setup