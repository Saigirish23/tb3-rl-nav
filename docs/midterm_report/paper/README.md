# IEEE Conference Mid-Term Report Package

This directory contains the editable LaTeX sources, generated high-resolution figures, standalone bibliography, and compiled PDF report for the mid-term progress milestone of the TurtleBot3 PPO LiDAR navigation project.

---

## Directory Structure

```
docs/midterm_report/paper/
├── main.tex                  # Complete editable IEEE conference LaTeX source
├── references.bib            # Standalone verified BibTeX bibliography
├── midterm_report.pdf        # Compiled publication-ready PDF report
├── generate_pdf.py           # Reproducible PDF build script (ReportLab)
├── verification_log.md       # Evidence audit verification log
├── README.md                 # Package instructions & compilation guide
└── figures/                  # Publication-quality figures (PDF and high-res PNG)
    ├── system_architecture.pdf
    ├── system_architecture.png
    ├── evaluation_results.pdf
    ├── evaluation_results.png
    ├── trajectory_metrics.pdf
    └── trajectory_metrics.png
```

---

## Authors & Project Metadata

- **Title:** *Sim-to-Real Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation: Mid-Term Progress Report*
- **Authors:**
  - **Saigirish S** — B24ME1066 — `b24me1066@iitj.ac.in` (Dept. of Mechanical Engineering)
  - **Siddhi Vinayak** — B24BB1036 — `b24bb1036@iitj.ac.in` (Dept. of Bioscience and Bioengineering)
- **Institution:** Indian Institute of Technology Jodhpur
- **Contribution Declaration:** *Both authors contributed equally to the project; their responsibilities were complementary.*
- **Course & Advisor Fields:** Administrative fields are left as standard IEEE conference placeholders (`[Course Name and Code]`, `[Faculty Advisor]`) to be specified upon submission.

---

## Compilation Instructions

### 1. LaTeX Compilation (Overleaf / Local TeX Live)
To compile the official IEEE two-column paper via standard LaTeX tools:
```bash
# Standard pdflatex + bibtex sequence
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
*Note:* The template requires `IEEEtran.cls` and standard TeX packages (`amsmath`, `graphicx`, `cite`, `booktabs`, `microtype`).

### 2. Standalone Python Build
If a TeX Live distribution is unavailable on the local machine, the identical publication-ready PDF can be regenerated with:
```bash
python3 generate_pdf.py
```
This produces `midterm_report.pdf` using vector and raster graphics exported directly from the audited project artifacts.
