#!/usr/bin/env bash
# Build the proposal, regardless of the caller's working directory.
set -euo pipefail

proposal_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$proposal_dir"

if ! command -v pdflatex >/dev/null 2>&1; then
    printf 'Error: pdflatex is required. Install a LaTeX distribution and retry.\n' >&2
    exit 1
fi

build_log="$proposal_dir/main.build.log"
: > "$build_log"

run_step() {
    if ! "$@" >> "$build_log" 2>&1; then
        printf 'Build failed: %s\n' "$*" >&2
        tail -n 40 "$build_log" >&2
        printf 'Full output: %s\n' "$build_log" >&2
        exit 1
    fi
}

printf 'Building proposal...\n'
run_step pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex

# Check the controller and section auxiliary files for citations.
shopt -s nullglob
aux_files=(main.aux sections/*.aux)
if grep -q '^\\citation{' "${aux_files[@]}"; then
    if ! command -v bibtex >/dev/null 2>&1; then
        printf 'Error: bibtex is required to render the proposal citations.\n' >&2
        exit 1
    fi
    run_step bibtex main
    run_step pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
fi

run_step pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
printf 'PDF: %s/main.pdf\nBuild log: %s\n' "$proposal_dir" "$build_log"
