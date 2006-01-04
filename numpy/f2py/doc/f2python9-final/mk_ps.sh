#!/bin/sh
cd src

test -f aerostructure.eps ||  convert ../aerostructure.jpg aerostructure.eps
test -f flow.eps || convert ../flow.jpg flow.eps
test -f structure.eps || convert ../structure.jpg structure.eps

latex python9.tex
latex python9.tex
latex python9.tex

dvips python9.dvi -o ../f2python9.ps
cd ..
gzip -f f2python9.ps
