#!/bin/bash
# IEEE T-IFS Article B Compilation Script
# Runs full LaTeX compilation with bibliography

echo "================================================================"
echo "IEEE T-IFS Article B - LaTeX Compilation Script"
echo "================================================================"
echo ""

# Check if main.tex exists
if [ ! -f "main.tex" ]; then
    echo "ERROR: main.tex not found in current directory"
    echo "Please run this script from the latex directory"
    exit 1
fi

# Clean previous build artifacts
echo "[1/5] Cleaning previous build artifacts..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot

# First pdflatex pass
echo "[2/5] Running pdflatex (pass 1)..."
pdflatex -interaction=nonstopmode main.tex > /tmp/pdflatex_pass1.log 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: pdflatex pass 1 completed with warnings (this is normal)"
fi

# Run bibtex
echo "[3/5] Running bibtex..."
bibtex main > /tmp/bibtex.log 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: bibtex completed with warnings"
    cat /tmp/bibtex.log
fi

# Second pdflatex pass
echo "[4/5] Running pdflatex (pass 2)..."
pdflatex -interaction=nonstopmode main.tex > /tmp/pdflatex_pass2.log 2>&1

# Third pdflatex pass (for cross-references)
echo "[5/5] Running pdflatex (pass 3)..."
pdflatex -interaction=nonstopmode main.tex > /tmp/pdflatex_pass3.log 2>&1

# Check if PDF was generated
if [ -f "main.pdf" ]; then
    echo ""
    echo "================================================================"
    echo "✓ Compilation SUCCESSFUL"
    echo "================================================================"
    echo ""

    # Get PDF info
    PAGES=$(pdfinfo main.pdf | grep "Pages:" | awk '{print $2}')
    SIZE=$(ls -lh main.pdf | awk '{print $5}')

    echo "Output file: main.pdf"
    echo "Pages:       $PAGES"
    echo "File size:   $SIZE"
    echo ""

    # Check for errors in final log
    ERRORS=$(grep -c "^!" /tmp/pdflatex_pass3.log)
    WARNINGS=$(grep -c "Warning" /tmp/pdflatex_pass3.log)

    echo "Errors:      $ERRORS"
    echo "Warnings:    $WARNINGS"
    echo ""

    if [ $ERRORS -gt 0 ]; then
        echo "⚠ ATTENTION: LaTeX errors detected in compilation"
        echo "Check /tmp/pdflatex_pass3.log for details"
    elif [ $WARNINGS -gt 10 ]; then
        echo "⚠ Note: $WARNINGS warnings detected (mostly citation warnings)"
        echo "Run full bibliography compilation to resolve"
    else
        echo "✓ Compilation clean - ready for review"
    fi

else
    echo ""
    echo "================================================================"
    echo "✗ Compilation FAILED"
    echo "================================================================"
    echo ""
    echo "Check logs in /tmp/ directory:"
    echo "  - /tmp/pdflatex_pass1.log"
    echo "  - /tmp/bibtex.log"
    echo "  - /tmp/pdflatex_pass2.log"
    echo "  - /tmp/pdflatex_pass3.log"
    exit 1
fi

echo ""
echo "================================================================"
echo "Next steps:"
echo "  - Review main.pdf"
echo "  - Check sections 7 and 8 (placeholder content)"
echo "  - Verify all citations in references.bib"
echo "  - Add figures/tables as needed"
echo "================================================================"
echo ""
