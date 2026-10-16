# Proposal blueprint

`main.tex` is the controller. Each of the four main sections has one `.tex` source and one generated `.aux` file in `sections/`. Subsections live inside their parent section file. All sections flow continuously, without forced page breaks.

The controller uses `\includesection`, a wrapper around `\include` that suppresses its automatic page breaks while retaining separate auxiliary files. Compile `main.tex` through `render.sh`; do not edit generated `.aux` files. The renderer checks the controller and section auxiliary files for citations before running BibTeX.

## Content files

- `sections/01-project-overview.tex`: title, team, motivation, and research question.
- `sections/02-research-paper-and-critical-analysis.tex`: key paper, summary, critique, extension, and related work.
- `sections/03-proposed-project-and-innovation.tex`: workflow, visualization/ML connection, and innovation.
- `sections/04-project-plan-and-evaluation.tex`: dataset, implementation, milestones, evaluation, risks, and references.

Blue bracketed text is a prompt to replace. The assignment requires a **4-page final PDF**; recheck the page count after editing. `main.pdf` is the compiled blueprint for review, not a submission-ready proposal.

## Decisions to make as a team

- Choose a target user and a specific task before selecting a visualization or model.
- Select and read one key paper; verify its methods and evaluation. Add two or three directly relevant supporting sources as space allows.
- Confirm dataset access, license, available labels, scale, and a fallback before committing to the topic.
- Define the minimum artifact, closest baseline, one testable difference, and a measurable success criterion.
- Choose a feasible evaluation and confirm roles and milestone dates. Assign the blank milestone leads together.
- Replace last-name, email, title, and team-number placeholders. Emails use `NAME_NETID@nyu.edu` and are not actual addresses.

## Build and preview

From the workspace root, with a LaTeX installation:

```sh
bash ml-visualization-project/proposal/render.sh
```

Or, from this directory:

```sh
./render.sh
```

The script works from any working directory and writes `main.pdf` beside `main.tex`. It runs BibTeX automatically when citations appear in the document or included sections, then reruns LaTeX to resolve references. Build output is saved to `main.build.log`; failures print the last 40 lines and return a nonzero exit status. The initial blueprint skips BibTeX because no papers have been selected.

In Overleaf, upload `main.tex`, `references.bib`, and the entire `sections/` folder, select `main.tex` as the main document, and recompile after adding real citations.

## Final submission checklist

- All prompts and placeholder emails replaced; title, team number, names, and addresses confirmed.
- Key paper identified, accurately summarized, critically analyzed, and connected to the proposed extension.
- Concrete artifact, research question, visualization/ML connection, and expected contribution explained.
- Dataset acquisition, implementation, relevant literature, evaluation baseline, measures, and feasibility specified.
- All cited sources verified; bibliography rendered; no unresolved citations or layout overflow.
- Final PDF visually checked and exactly **4 pages**, including references under the original conservative page budget.
- Rename the completed PDF to the actual team number, such as `Team5.pdf`; do not submit a file with a placeholder team number.
- Submit **one PDF per team on Brightspace by Tuesday, October 20, 2026**. No submission time was provided; confirm it on Brightspace.
