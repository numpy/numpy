#!/bin/sh
cd src

test -f aerostructure.eps ||  convert ../aerostructure.jpg aerostructure.eps
test -f flow.eps || convert ../flow.jpg flow.eps
test -f structure.eps || convert ../structure.jpg structure.eps

latex python9.tex
latex python9.tex
latex python9.tex

test `which tth` && cat python9.tex | sed -e "s/{{}\\\verb@/\\\texttt{/g" | sed -e "s/@{}}/}/g" | tth -Lpython9 -i > ../f2python9.html
cd ..
