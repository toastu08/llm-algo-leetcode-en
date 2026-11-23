# AGENTS.md — English Translation Workstream

> **Purpose:** This file governs the English translation of the datawhale
> `llm-algo-leetcode` tutorial repository. It applies only to translation work.
> If you are working on unrelated code or repository maintenance, follow the
> normal project conventions instead.

## 1. Read this first

Before touching any file, read these documents **in order**:

1. `AGENTS.md` — this file: rules, workflow, and prohibitions.
2. `SPEC.md` — exact scope, milestones, batches, and file inventories.
3. `PROGRESS.md` — current status and the files still outstanding.
4. `docs/terminology.md` — the canonical Chinese→English terminology reference.

Do not start translating from memory. The terminology table is the source of
truth for technical terms, and `SPEC.md` is the source of truth for what is in
scope.

## 2. Mission

Translate the tutorial content from Chinese into faithful, natural technical
English while preserving:

- the original meaning, caveats, and nuance;
- the tutorial structure (sections, questions, verification steps, tables);
- all code behavior, identifiers, APIs, and metadata;
- all links, anchors, image paths, and asset references.

This is a **translation project, not an editing or improvement project**. If
something looks outdated, wrong, awkward, or inconsistent, translate it
faithfully and record the issue in `PROGRESS.md` under “Content issues”. Do not
fix it.

## 3. Scope

### 3.1 In scope

- Part source trees: `00_Prerequisites/`, `01_Hardware_Math_and_Systems/`,
  `02_PyTorch_Algorithms/`, `03_Triton_Kernels/`,
  `04_CUDA_and_System_Optimization/` — top-level `.md` and `.ipynb` files only.
- `05_Integrated_Projects_and_Decisions/` — all `.md` and `.ipynb` files.
- `topic_discussion/` — all `.md` files.
- `team_study/` — all `.md` files.
- The docs-only pages listed in `SPEC.md` (for example `docs/guide.md`,
  `docs/maintenance.md`, `docs/verification/*.md`).
- The optional development/environment pages listed in `SPEC.md` (for example
  `cnb/README.md`, `tools/README.md`).

### 3.2 Out of scope

- `README.md` (excluded by decision).
- `docs/index.md` (generated from `README.md`; do not edit).
- All generated docs mirrors:
  - `docs/00_Prerequisites/` … `docs/04_CUDA_and_System_Optimization/`
  - `docs/topic_discussion/`
  - `docs/team_study/`
  - all `docs/05_Integrated_Projects_and_Decisions/` mirror pages
- Images, SVGs, PNGs, and other binary assets. Translate image **alt text** and
  captions in Markdown, but never edit the image files themselves.
- `.ipynb_checkpoints/`, `.venv/`, `node_modules/`, caches, and build output.
- Code and configuration: `tools/*.py`, `verify.py`, `.github/`, `src/`,
  `docs/.vitepress/`, `package.json`, `package-lock.json`, `pyproject.toml`,
  `uv.lock`, `requirements*`, `environment.yml`, `.cnb.yml`.
- Benchmark model cards under `benchmarks/artifacts/` (already English/zero CJK).
- Any file whose status in `SPEC.md` is “N/A” or “derived”.

### 3.3 Source of truth vs generated docs

The repository keeps tutorial sources in the Part directories and publishes
generated mirrors under `docs/`.

**Translate the sources, then regenerate the mirrors.** Do not hand-edit a
generated mirror: the next regeneration overwrites it.

The generators are:

| Tool | What it produces |
| --- | --- |
| `tools/convert_notebook.py` | Notebook/prose mirrors for Parts 00–04 |
| `tools/sync_docs_navigation.py` | Intro/group pages, `topic_discussion/`, `team_study/` |
| `tools/sync_docs_index.py` | `docs/index.md` from `README.md` — **do not run** |

`docs/05_Integrated_Projects_and_Decisions/` is only partially covered by the
sync tool. Its sub-page `index.md` files and notebook mirrors are produced once
during final QA (see `SPEC.md` §6). Do not translate those mirror copies
directly.

## 4. Golden rules

1. **Translation only.** Do not improve, correct, refactor, modernize, expand,
   shorten, or reinterpret the content.
2. **No behavior changes.** Do not change tutorial logic, identifiers, imports,
   numeric values, assertions, cell metadata, tags, or execution structure.
   Comments and docstrings may be translated per §7; that is the only allowed
   edit inside code cells.
3. **Preserve structure.** Keep headings, numbering, tables, callouts, lists,
   blank-line structure, separators, and code fences.
4. **No omissions.** Do not summarize, merge, or delete sentences. Translate
   every Chinese prose passage.
5. **No renaming.** Do not rename files, directories, notebook cells, or assets.
6. **Do not translate generated docs directly.** Fix the source and regenerate.
7. **Do not translate `README.md` or `docs/index.md`.**
8. **Do not touch images or binaries.**
9. **When in doubt, translate literally** and note the ambiguity in
   `PROGRESS.md`.

## 5. Terminology

Read `docs/terminology.md` before every batch. It lists canonical English names
and abbreviations.

- Use the **English / 缩写** column as the canonical term.
- Keep library names, backend names, model names, API names, protocol names,
  identifiers, JSON keys, file paths, URLs, artifact IDs, and notebook tags in
  English.
- Do not carry the Chinese `中文（English）` first-occurrence pattern into
  English prose. Write the English term directly; add the original Chinese in
  parentheses only when it genuinely helps traceability (for example, a quoted
  section title or a term that has no stable English equivalent).
- Keep established abbreviations such as KV Cache, TTFT, TPOT, DPO, GRPO, MoE,
  GQA, MLA, RoPE, RMSNorm, and PagedAttention.
- If a new technical term appears that is not in the table, use a natural
  English equivalent, be consistent within the batch, and add it to the
  “Glossary additions” section of `PROGRESS.md`.
- When translating `docs/terminology.md` itself in M9, **preserve the Chinese
  and English term columns and the table structure**. Translate only the
  explanatory prose. Do not drop the Chinese column; it is the mapping
  reference for the whole project.

## 6. Notebook workflow (python-notebook MCP)

Use the `python-notebook` MCP for reading and editing notebooks. It preserves
Jupyter JSON structure and cell metadata.

### 6.1 Initialize once per session

Call `initialize_workspace` with the **absolute** repository path before any
other notebook tool:

```text
initialize_workspace(directory="/home/toastu/llm-algo-leetcode")
```

In Code Mode, call it as:

```js
await tools["python-notebook"].initialize_workspace({
  directory: "/home/toastu/llm-algo-leetcode"
});
```

### 6.2 Available tools

- `list_notebooks` — inventory
- `read_notebook` — read the whole notebook
- `read_cell` — read one cell
- `read_cell_output` / `read_notebook_outputs` — inspect outputs
- `edit_cell` — replace one cell's source
- `add_cell` — do **not** use for translation
- `create_notebook` — do **not** use for translation

### 6.3 Editing rules

- Read the cell before editing it.
- Use `edit_cell` to replace the **entire** cell source with the translated
  source. Do not attempt partial string replacement through raw JSON.
- Never add, delete, reorder, split, merge, or retype cells for translation.
- Preserve cell `id`, `metadata`, `tags`, `execution_count`, and `outputs`.
- Do not execute notebooks. Preserve their outputs as-is.
- If the MCP is unavailable, edit the `.ipynb` with Python/`json`/`nbformat`,
  then validate the JSON and schema. Never use a blind text replace on the JSON.
- Leave notebook outputs unchanged. Only 22 notebooks contain Chinese output
  text; record them in `PROGRESS.md` rather than re-running or rewriting them.

## 7. Code-cell policy

The agreed policy is **comments + docstrings only**:

- Translate Chinese **comments**.
- Translate the human-readable prose inside module/class/function **docstrings**.
- Leave every other string literal unchanged, including `print()`, `assert`,
  `raise`, logging messages, dictionary keys, column names, file paths, config
  values, expected outputs, and test data.
- Leave code examples and expected output inside docstrings unchanged.
- Preserve directive comments verbatim: `#!`, encoding declarations,
  `# type:`, `# noqa`, `# pragma:`, `# fmt:`, `# isort:`, `# pyright:`,
  `# ruff:`, `# nosec`.
- Do not reformat code, change indentation, reorder lines, or rename variables.
- Do not translate Chinese comments that are actually data (for example, a
  Chinese string inside a list literal). If it is a string literal, leave it.
- Never translate `STOP HERE`, `TODO`, tag names, or other structural markers.

Before and after editing a batch, compare the code-cell ASTs to catch accidental
logic changes. A change is acceptable only if it is confined to comments,
docstrings, or whitespace.

## 8. Markdown policy

Translate all Chinese prose, including:

- body paragraphs, bullet lists, numbered lists;
- table cells and headers;
- blockquotes and callouts;
- `<details>` / `<summary>` visible text;
- image alt text and figure captions;
- Mermaid node/edge labels;
- prose inside fenced code blocks (apply the code-cell policy to comments and
  docstrings only).

Preserve:

- heading levels and numbering;
- pipe-table structure and row/column counts;
- blank lines and `---` separators;
- HTML tags and attributes;
- `<a id="...">` anchors;
- code fences, inline code, formulas, math, paths, URLs, and IDs;
- image paths and link targets.

### 8.1 Headings

- Translate heading text but keep the level, numbering, and hierarchy.
- Keep `Q1`, `Q2`, `Step 1`, `Task 0`, etc. as structural markers; translate the
  descriptive text around them. For example:
  - `## Q1：如何读取...` → `## Q1: How do you read...`
  - `## Q1 验证：...` → `## Q1 verification: ...`
  - `### Step 1: 定义...` → `### Step 1: Define...`
- When a heading is already bilingual (`English | 中文`), keep the English half
  and drop the Chinese half, unless the Chinese half carries distinct meaning
  that must be preserved.
- When a translated heading had inbound anchor links, add an explicit HTML
  anchor before it to preserve the old link. For example:

  ```md
  <a id="基础与诊断入口"></a>
  ## Foundations and Diagnosis
  ```

  Search the repository for the old heading slug before changing it.

### 8.2 Tables

Translate every cell, including headers. Keep alignment markers and the exact
row/column structure. Do not drop or add rows for “clarity”.

### 8.3 Links

- Translate the link label; leave the target unchanged.
- Do **not** rewrite `.ipynb` links to `.md` in the source tree. The docs
  generator performs that rewrite for the mirror.
- Preserve relative paths (`./`, `../`) exactly.
- If a heading anchor changes, update every internal link to the old anchor and
  add the compatibility anchor described above.

### 8.4 Images and diagrams

- Keep image paths unchanged.
- Translate alt text and captions.
- Do not edit SVGs/PNGs. The diagrams will remain Chinese for now; this is a
  known, accepted limitation. Record it in `PROGRESS.md` if a page depends
  heavily on diagram text.

### 8.5 Math and code fences

- Do not translate LaTeX/MathJax structure. Translate prose around it.
- If `\text{中文}` appears, translate the text inside `\text{}` while preserving
  the LaTeX.
- Apply §7 to code fences: comments and docstrings only, other strings
  untouched.

## 9. H1 policy

### 9.1 Numbered notebooks (Parts 00–04)

For numbered notebooks in Parts 00–04, the first H1 must remain exactly in the
form:

```md
# 05. PyTorch Tensor Fundamentals | PyTorch 张量基础
```

**Do not rewrite, translate, shorten, or remove that numbered H1 line.** The
repository validator `verify.py` requires the `# NN. ` prefix and the ` | `
separator, and `tools/convert_notebook.py` relies on the Chinese half of the
title when it regenerates the docs mirror.

### 9.2 Part 05 notebooks

Part 05 project notebooks are not numbered and are not covered by the
`verify.py` H1 check. Translate their H1s to English:

- If the H1 is bilingual, keep the English half and drop the Chinese half.
- If it is Chinese-only, produce a natural English title.
- Do not add a numeric prefix.

### 9.3 Prose pages

For prose pages (`intro.md`, group pages, topic pages), H1s may be translated.
If the H1 is bilingual, keep the English half and drop the Chinese half.

## 10. Per-file workflow

1. Confirm the file is in scope and in your assigned batch (`SPEC.md`).
2. Read the whole file before editing.
3. Translate all in-scope prose.
4. Apply the code-cell policy.
5. Check for unintended Chinese prose left behind (see §12).
6. Validate notebook JSON/schema or Markdown structure.
7. Mark the file `[x]` in `PROGRESS.md` and add a short note if needed.
8. Do not regenerate docs until the batch is complete.

## 11. Batch and milestone workflow

1. Read the milestone/batch entry in `SPEC.md` and the current state in
   `PROGRESS.md`.
2. Translate every file in the batch.
3. Run the batch checks (§12).
4. Regenerate the affected mirrors:
   - For Parts 00–04 prose/notebooks:
     `python tools/convert_notebook.py --dir <part>` and
     `python tools/sync_docs_navigation.py --part <part>`.
   - For `topic_discussion`: `python tools/sync_docs_navigation.py --topic <topic> --prune`.
   - For `team_study`: `python tools/sync_docs_navigation.py --part team_study`.
   - For `docs/05` sub-pages: final QA only (see `SPEC.md` §6).
5. Run link/mirror checks.
6. Update the dashboard, session log, and notes in `PROGRESS.md`.
7. Commit the batch on the translation branch (one commit per batch recommended).

Do **not** run `tools/sync_docs_index.py`; it overwrites `docs/index.md` from
the excluded `README.md`.

## 12. Validation

### 12.1 Notebook JSON and schema

```bash
python -m json.tool path/to/notebook.ipynb > /dev/null
python -c "import nbformat,sys; nbformat.validate(nbformat.read(sys.argv[1], as_version=4))" path/to/notebook.ipynb
```

### 12.2 Code-cell AST comparison

For each translated notebook, compare the current code cells with the original
from Git:

```bash
# On the translation branch, main holds the pre-translation baseline.
git show main:path/to/notebook.ipynb > /tmp/original.ipynb
```

If you are working directly on `main`, replace `main` with the pre-translation
commit hash.

Parse both with `ast` per code cell. A safe translation changes only comments,
docstrings, or whitespace. If the AST differs elsewhere, revert and redo the
cell.

### 12.3 Residual Chinese scan

Scan for Chinese characters in Markdown cells and prose. Review every hit and
classify it:

- **Allowed:** numbered notebook H1 title lines in Parts 00–04; intentional
  code strings; terminology examples; preserved compatibility anchors;
  generated cloud-run boilerplate.
- **Not allowed:** untranslated body prose, table cells, captions, alt text,
  headings, Mermaid labels.

Record the residual counts in the batch notes.

### 12.4 Links and mirrors

```bash
python tools/check_source_docs_mirror.py
python tools/check_chapter_links.py
python tools/check_docs_links.py   # requires docs npm dependencies
```

`check_docs_links.py` ignores anchors, so manually verify any heading anchor you
changed.

### 12.5 Known validator false positives

Several repository validators encode the **Chinese** content contract. After
translation they may report failures that are not real defects:

| Validator | Chinese expectation |
| --- | --- |
| `verify.py` | Numbered bilingual H1 in Parts 00–04 (`# NN. English \| Chinese`) — avoid by following §9 |
| `tools/audit_chapter0_1_notebooks.py` | `本节导读`, `前置阅读`, `相关阅读` |
| `tools/validate_notebook_format.py` | `## 测试` |
| `tools/test_framework_conformance.py` | `## 框架一致性核对（可选）` |
| `tools/check_mechanism_step4_tables.py` | Chinese Step 4 table header |

Do not edit these tools to make them pass. Treat their heading/title failures as
expected; investigate only failures that indicate broken JSON, broken links, or
changed code logic.

## 13. Definition of done

A file is done only when:

- [ ] Every Chinese prose passage is translated.
- [ ] The code-cell policy has been applied.
- [ ] Structure, tables, links, anchors, and image paths are preserved.
- [ ] No unintended Chinese prose remains.
- [ ] Notebook JSON/schema is valid; cell structure and metadata are unchanged.
- [ ] Code-cell AST comparison shows no logic change.
- [ ] The file is marked `[x]` in `PROGRESS.md`.

A batch is done only when:

- [ ] Every file in the batch meets the file-level definition.
- [ ] Residual-CJK scan has been reviewed.
- [ ] Generated mirrors for the affected part have been rebuilt.
- [ ] Link/mirror checks have been run and results recorded.
- [ ] The dashboard and session log in `PROGRESS.md` are updated.

## 14. Subagent rules

When delegating a batch to a subagent:

- Give it the milestone ID, batch ID, and exact file list.
- Tell it to read `AGENTS.md`, `SPEC.md`, `PROGRESS.md`, and
  `docs/terminology.md` first.
- Assign disjoint files. Never let two agents edit the same file.
- Require it to update `PROGRESS.md` and report residual-CJK counts.
- Forbid edits to tools, tests, configs, images, `README.md`, and
  `docs/index.md`.
- Ask it to report any content issue instead of fixing it.

## 15. Branch and commit hygiene

- Recommended branch: `translation/en`.
- Commit one batch at a time, for example:
  `docs(en): translate Part 02 batch 02-3 (26–36)`.
- Do not commit the user's untracked local files (`.python-version`,
  `pyproject.toml`, `src/`, `uv.lock`).
- Keep generated docs and source changes in the same commit so the mirror stays
  consistent.
