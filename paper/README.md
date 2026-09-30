# Fortschrittsbericht 1

`main.tex` uses the MVSR paper template (IEEE conference `IEEEtran` class,
3-page single-author paper). It compiles with any standard LaTeX
distribution or on Overleaf — `IEEEtran.cls` and `IEEEtranS.bst` ship with
TeX Live/MiKTeX/Overleaf by default and are not vendored here.

Build locally:

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Status (Fortschrittsbericht 1)

- Outline of all IMRAD chapters with bullet-point planned content: done
  (Sections II–V).
- Fully worded scientific contribution, problem statement, and research
  question: done (Section I, Introduction).
- Still open for later progress reports: full prose for Sections II–IV
  (Fortschrittsbericht 2/3), figures (mesh renders, mask examples), and
  fitting the paper to exactly 3 pages.
