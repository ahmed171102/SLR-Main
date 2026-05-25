# Eshara — Thesis LaTeX Skeleton

This folder is a complete LaTeX scaffold for the FYP report, matching the
AASTMT *FYP Template Final - Individual Report* formatting requirements
(Times 12 pt body, 1.5 line spacing, A4, ~1-inch margins, IEEE-style
numbered citations, chapter-prefixed figure/table/equation numbering).

## Folder layout

```
latex/
├── main.tex                # master file — compile this
├── bibliography.bib        # IEEE numeric references (add more as needed)
├── chapters/
│   ├── acronyms.tex
│   ├── 00_titlepage.tex
│   ├── 00_declaration.tex
│   ├── 00_dedication.tex
│   ├── 00_acknowledgments.tex
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_literature_review.tex
│   ├── 03_methodology.tex
│   ├── 04_implementation.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   ├── 07_conclusion.tex
│   ├── 08_future_work.tex
│   └── 99_appendix.tex
└── figures/                # PNG/PDF figures referenced from chapters
```

The Mermaid diagrams rendered into `Current Thesis/figures/*.png` are
referenced by relative path; the `\graphicspath` directive in `main.tex`
picks them up automatically.

## How to build

### One-shot

```bash
latexmk -pdf main.tex
```

### Manual (in order)

```bash
pdflatex  main
bibtex    main
pdflatex  main
pdflatex  main
makeglossaries main
pdflatex  main
```

### Recommended installs

- **TeX Live 2024** (Linux/macOS) or **MiKTeX 24+** (Windows)
- `latexmk` (bundled with both distributions)
- The `mathptmx`, `titlesec`, `biblatex`, `glossaries`, `cleveref`,
  `tabularx`, `siunitx`, and `hyperref` packages — all included in a
  full TeX Live install.

### One-line install on Windows (Chocolatey)

```powershell
choco install miktex -y
```

## Filling in the placeholders

Every chapter contains `XX.X %` style placeholders to be replaced with
your final experimental numbers, plus `TODO` comments next to figure or
table inserts that still need data. Search for `XX` and `TODO` to find
them all:

```bash
grep -nR "XX" chapters/
grep -nR "TODO" chapters/
```

## Updating the bibliography

1. Add a new `@inproceedings{...}` or `@article{...}` block to
   `bibliography.bib`.
2. Cite it in any chapter with `\cite{key}`.
3. Re-run `bibtex main` then `pdflatex main` twice.

## Updating the acronym list

Add a `\newacronym{...}` line in `chapters/acronyms.tex` and use
`\gls{key}` in body text. The list at the front of the document
regenerates automatically with `makeglossaries main`.

## Word version

If your supervisor requires a Word submission, the easiest path is:

```bash
pandoc main.tex -o eshara_thesis.docx --bibliography bibliography.bib
```

(There will be some manual cleanup of figure/table sizes and equation
numbers in Word, but the content transfers cleanly.)
