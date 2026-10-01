# Generated tables and figures

`make_figures.py` writes the LaTeX tables into `tab/` and the figures into
`fig/`. The manuscript includes them with

```latex
\input{tab/t1_conv.tex}
\includegraphics[width=\linewidth]{fig/conv1.pdf}
```

so dropping these two directories next to the `.tex` file is all that is
needed. Nothing here is under version control: the files are regenerated from
`results/` in about a minute. The table of the top-level README says which
float of the paper each file corresponds to.
