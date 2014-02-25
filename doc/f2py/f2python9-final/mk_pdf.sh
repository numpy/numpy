#!/bin/sh
cd src

test -f aerostructure.pdf ||  convert ../aerostructure.jpg aerostructure.pdf
test -f flow.pdf || convert ../flow.jpg flow.pdf
test -f structure.pdf || convert ../structure.jpg structure.pdf

cat python9.tex | sed -e "s/eps,/pdf,/g" > python9pdf.tex
pdflatex python9pdf.tex
pdflatex python9pdf.tex
pdflatex python9pdf.tex

mv python9pdf.pdf ../f2python9.pdf