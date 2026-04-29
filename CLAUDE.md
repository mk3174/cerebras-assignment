# CLAUDE.md — guidance for AI assistants on this repo

This file is the load-bearing context for any AI helping on this challenge.
Read it first. Authoritative spec is `SPEC.md`; this file is the onramp.

## What this repo is

Cerebras Kernel Challenge: implement Top-K k-NN on the Wafer-Scale Engine
in **CSL** (Cerebras's tile language). Given a query `q` (fp32, shape `d`)
and a database `D` (fp32, `N×d`) sharded across a `P×P` PE grid, return
the `K` rows of `D` closest to `q` by **squared** Euclidean distance,
sorted ascending, with deterministic tie-breaking (smaller original index
wins on equal distance).

72-hour clock starts when the candidate opens the packet. Log the start
time in `DESIGN.md` (per `README.md`).

## Repo layout

```
.
├── README.md             # Onramp + run instructions
├── SPEC.md               # Authoritative spec — grading, constraints, deliverables
├── reference.py          # NumPy oracle. `python reference.py` prints expected top-3 per case
├── tests/
│   └── test_correctness.py  # Grader. Compiles + runs candidate kernel for 6 cases
├── cycles.json           # Per-case reference cycle counts (currently all null → perf gate waived)
├── src-starter/
│   └── README.md         # Tutorial reading order + common pitfalls
├── docker/
│   ├── Dockerfile        # Ubuntu 22.04 + Apptainer + SDK
│   ├── README.md         # Docker build + run instructions (Mac/Windows path)
│   └── .gitignore        # Excludes ./sdk and SDK tarballs
├── requirements.txt      # numpy + pytest (host-side venv)
└── .venv/                # Local venv for running reference.py / pytest on host
```

## Deliverables (must exist at the dir the grader is pointed at)

- `layout.csl` — wafer layout, color/route declarations
- `pe_program.csl` (and/or others) — PE kernel(s)
- `run.py` — host: memcpy `D` and `q`, launch, read back, compare to oracle, print `PASS: <case>`. Must accept `--name <build_dir>` and `--case <case_name>`.
- `commands.sh` — one-shot `cslc` + `cs_python` for the baseline case
- `DESIGN.md` — exactly one page, ~400–600 words. See `SPEC.md §5`.

The grader (`tests/test_correctness.py`) is invoked with
`--submission=<dir>` and runs each case as:

```bash
cslc --arch=wse2 layout.csl \
  --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:<P>,d_dim:<d>,rows_per_pe:<rpp>,K:<K> \
  --memcpy --channels=1 -o <build_dir>
cs_python run.py --name <build_dir> --case <case_name>
```

Per-case params (from `tests/test_correctness.py`):

| case        | P | d  | rows_per_pe | K   |
|-------------|---|----|-------------|-----|
| baseline    | 4 | 32 | 128         | 16  |
| k_eq_1      | 2 | 32 | 256         | 1   |
| k_large     | 2 | 16 | 256         | 256 |
| uneven      | 4 | 32 | 64          | 16  |
| all_equal   | 2 | 16 | 256         | 16  |
| duplicates  | 2 | 16 | 256         | 8   |

## Hard rules (will fail grader if violated)

- Return **squared** L2 distances (`||x-q||²`). NOT Euclidean. The oracle returns squared and tolerance is `atol=1e-3, rtol=1e-3`.
- Tie-break by smaller original index. Tested by `all_equal` and `duplicates` cases.
- **Deterministic** — same inputs → bit-identical output across runs. Beware fabric arrival order.
- Each PE has ~**48 KB SRAM**. Oversized buffers fail at link, not compile.
- `cslc` must be invoked with `--memcpy --channels=1` (SdkRuntime). The default `CSELFRunner` is deprecated and will reject.
- `layout.csl` and `run.py` must sit in the same directory the grader points at.

## Toolchain

The Cerebras SDK runs only on **Linux + Apptainer/Singularity**. Available paths:

- **Linux native**: install Apptainer, extract SDK, put on PATH.
- **Mac/Windows via Docker**: build `docker/` image (it bundles Apptainer + SDK). See `docker/README.md`.
- **Cloud Linux VM** (Render $50 promo `RENDER-CODETV`) or **GitHub Codespaces**.

This dev machine is **Apple Silicon (arm64)**. The SDK is x86_64-only, so under Docker:

- Build with `--platform=linux/amd64`.
- **Turn OFF Rosetta x86_64 emulation in Docker Desktop** — the compiler uses AVX2 which Rosetta doesn't translate (crashes with `Illegal instruction`, exit 132). QEMU is ~3–5× slower but works. See `docker/README.md`.

Once the SDK is on PATH (native or in container), the binaries you'll use:

- `cslc` — CSL compiler
- `cs_python` — Python wrapped to find the SDK runtime libs
- `csdb` — cycle-count debugger / profiler (used for the perf gate)
- `sdk_debug_shell` — interactive

## Local (host) Python env

Already set up: `.venv/` with `numpy` + `pytest`. Used to run `reference.py` and (when run from a Linux/container) `pytest`. On Mac you can only run `reference.py` here, not the full grader.

```bash
.venv/bin/python reference.py     # prints expected top-3 per case
```

## Tutorials to read before writing CSL

From `src-starter/README.md`, in order:

1. `examples/tutorials/gemv-00-basic-syntax`
2. `examples/tutorials/gemv-05-multiple-pes` (multi-PE + memcpy)
3. `examples/tutorials/gemv-06-routes-1` … `gemv-08-routes-3` (routing)
4. `examples/tutorials/topic-11-collectives` (broadcast/gather/reduce)
5. `examples/tutorials/topic-05-sentinels`
6. `examples/benchmarks/gemv-collectives_2d` ← best template for P×P grid
7. `examples/benchmarks/histogram-torus` (wavelet bit-packing) — only if needed

Confirm the toolchain works by compiling and running `gemv-collectives_2d` end-to-end (`bash commands_wse2.sh` — should print `SUCCESS`) **before** writing your own kernel.

## Common pitfalls (from `src-starter/README.md` and SPEC)

- **Task IDs share a 32-slot namespace with colors.** Many IDs are reserved by memcpy and collectives. Check the color-map comment in `topic-11-collectives/layout.csl` before picking task IDs.
- **Collective callbacks fire on every PE in the group**, not just the destination. State machines must handle that.
- **Memcpy/collectives layout requires** `--memcpy --channels=1` at compile time.
- **Sqrt vs squared**: returning `sqrt(d²)` will pass tiny tests then fail at scale (numeric tolerance). Stay in squared L2.
- **Determinism**: don't merge top-K state from a `recv` callback in arrival order without a deterministic key (use lex `(dist, idx)`).

## Grading recap (`SPEC.md §4`, 100 pts)

- 25 baseline correctness · 25 edge-case correctness · 10 determinism · 20 perf · 15 design memo · 5 code clarity.
- Failing baseline correctness → 0 overall.
- Perf gate (`csdb` cycles ≤ 1.3× reference) currently waived (`cycles.json` all null) but optimize anyway — DESIGN.md is graded on perf reasoning regardless.

## Workflow conventions for AI helpers

- Edit existing files; don't create new top-level docs unless asked.
- Keep CSL files at repo root once written (per submission layout requirement). The starter dir is `src-starter/` but the grader is pointed at one dir; flatten before submission.
- Don't commit the SDK tarball or `docker/sdk/` (already gitignored).
- Don't share this challenge externally or ask current Cerebras employees for help (per `SPEC.md §7`).
- Use AI freely — but the candidate must understand every line. The DESIGN.md memo is the anti-plagiarism gate; vacuous memos are caught in the onsite.

## Version note

`README.md` links to **Cerebras SDK 2.10.0** as the current download. `docker/Dockerfile` and `docker/README.md` reference **SDK 1.4.0** filenames. If the downloaded tarball is named `Cerebras-SDK-2.10.0-*.tar.gz`, adjust the `tar`/`mv` lines in `docker/README.md` accordingly (the Dockerfile only cares that `./sdk/` exists with the right binaries inside).
