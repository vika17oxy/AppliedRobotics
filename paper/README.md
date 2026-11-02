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

## Status

- Full paper (3 pages, IEEE): introduction, state of the art, methods, results, summary — done.
- Results: view-count study (truck), real phone photos (figurine; why masked 3DGS still keeps background
  Gaussians and how the silhouette-based exporter fixes thin parts), CAD evaluation on a printed dragon
  page holder (Chamfer 1.71 mm from 21 photos).
- Figures in `figures/` (generated from the STLs in `../results/`); compiled `main.pdf` included.
