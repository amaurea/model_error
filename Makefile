opts=-halt-on-error -draftmode
main.pdf: main.tex
	pdflatex $< $(opts) && bibtex main && pdflatex $< $(opts) && pdflatex $<
	#pdflatex $<
clean:
	rm -r *.log *.aux *.blg *.out *.bbl

.PHONY: main.pdf clean
