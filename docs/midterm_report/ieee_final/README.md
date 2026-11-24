# IEEE Midterm Progress Report Package (Official IEEEtran Class)

This directory contains the completely audited and corrected IEEE conference-style midterm progress report, formatted strictly in accordance with IEEE conference standards.

---

## Directory Contents

```
docs/midterm_report/ieee_final/
├── main.tex                    # Editable LaTeX source using \documentclass[conference]{IEEEtran}
├── IEEEtran.cls                # Official IEEE conference LaTeX document class
├── references.bib              # Verified BibTeX bibliography including Marathon 2 Nav2 citation
├── midterm_report.pdf          # 5-page publication-quality PDF
├── generate_pdf.py             # Standalone Python PDF generator with strict IEEE layout & typography
├── AUDIT_SUMMARY.md            # Comprehensive audit summary of all verifications and calculations
├── UNVERIFIED_CLAIMS.md        # Documented items requiring author confirmation
├── README.md                   # This compilation and structure guide
└── figures/                    # Vector PDF and 300-DPI raster PNG figures
    ├── fig1_system_architecture.pdf
    ├── fig1_system_architecture.png
    ├── fig2_evaluation_results.pdf
    ├── fig2_evaluation_results.png
    ├── fig3_trajectory_metrics.pdf
    └── fig3_trajectory_metrics.png
```

---

## Compilation Instructions

### 1. LaTeX Compilation (Overleaf / Local TeX Live)
This directory includes the official `IEEEtran.cls` file, making it completely self-contained. To compile via standard tools:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 2. Standalone Python Build
To regenerate the exact publication-quality PDF in environments without TeX Live:
```bash
python3 generate_pdf.py
```
This produces `midterm_report.pdf` using vector figures and exact IEEE formatting standards.
