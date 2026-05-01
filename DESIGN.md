# Top-K kNN — Design memo

**Started:** 2026-04-29 (per `README.md` 72-hour clock).
**Target:** WSE-2, `P × P` PE grid, 6 grader cases (`tests/test_correctness.py`).

## 1. Routing topology

`<collectives_2d>` only — no custom routes. Colors `0,1` (x-dim) / `4,5` (y-dim); collective entrypoints task IDs `14–17`. Two local task IDs (`10`, `11`) form a callback chain: `kernel → mpi_x.gather → task_merge_x → mpi_y.gather → task_merge_y`. q is **replicated by the host** to every PE via H2D — kernel starts directly at distance compute (no fabric broadcast).

```
              px=0      px=1      px=2      px=3
   py=0   ● ◀ D2H       ●         ●         ●     ◀── mpi_x.gather(root=0) per row
          ▲             │         │         │         (P PEs → col 0, westward)
          │
   py=1   ●             ●         ●         ●     ◀──
          ▲                                              mpi_y.gather(root=0)
          │                                              along col 0 ONLY,
   py=2   ●             ●         ●         ●     ◀──    guarded `if (px == 0)`
          ▲                                              (avoids P-1 wasted gathers)
          │
   py=3   ●             ●         ●         ●     ◀──
```

Wavelet payloads: each `(dist, idx)` pair = `2` u32 (`bitcast<u32>(dist)`, `idx`). Per-PE gather chunk = `2K` u32. q never traverses the fabric.

## 2. Local top-K algorithm

Per row: `R·d` scalar `sub/mul/fma` → squared L2, paired with global index `(py·P + px)·R + i`.

Selection: **size-K max-heap, lazy replace-root, heap-sort on extract**. Insert is `O(log K)`; only inserts when `(dist, idx) <_lex root`. Heap orders by `dist` descending (root = worst); ties keep the **larger idx as the eviction victim**, preserving the spec's "smaller idx wins" output tie-break. Heap-extract emits K pairs ascending.

After the row-gather, `(0, py)` runs a **P-way cursor merge** over P sorted lists in `gather_recv`: pick lex-min head, advance its cursor, repeat. `O(P · K)`. Same routine reused at `(0, 0)` after col-gather.

| Stage | Cost | Worst case (`k_large`, P=2, R=K=256) |
|---|---|---|
| local distance + heap | `R·d + (R+K)·log₂K` | `4 096 + 4 096 = 8 192` |
| row-merge at `(0, py)` | `P · K` | `512` |
| col-merge at `(0, 0)` | `P · K` | `512` |

**Cycles per element**, scalar dominant loops:
- distance inner ≈ **4 cyc/elem** (sub, mul, fma) → `4·d` cyc/row.
- heap insert ≈ **6 cyc × log₂K levels**.
- merge head-pick ≈ **3 cyc / cursor**.

Bottleneck PE: `~9 K cyc` on `k_large`, `~17 K cyc` on baseline — compute-bound in both. Per-PE buffers ≤ `~26 KB`; well under the 48 KB ceiling.

## 3. Fabric bandwidth

Worst per-edge wavelet count during gather: `(P-1)·K·2 = 1 024 u32` for `k_large`. q-broadcast = 0 (host-replicated). Total per-edge ≈ 1 K wavelets per run — orders of magnitude under per-cycle saturation.

**Bottleneck = compute on every case.** Collective setup is hundreds of cycles per gather; the `~16 K`-cycle distance loop dominates. To be confirmed with `csdb` once `cycles.json` is non-null.

## 4. Tie-break (determinism proof)

Every comparison is lex `(dist:f32, idx:u32)`. Global indices `(py·P + px)·R + i` are unique across the padded grid → `(dist, idx)` is a **total order**; every subset has a unique sort.

The heap's eviction rule (larger idx out first on ties) leaves the K *smallest* `(dist, idx)` pairs in the heap regardless of insertion order; heap-extract walks them out ascending, so `topk_send` is bit-identical per shard run-to-run.

`<collectives_2d>` concatenates gather inputs by source-PE position deterministically → `gather_recv` is bit-identical. P-way cursor merge picks lex-min and is order-deterministic. **Final K pairs at `(0, 0)` are bit-identical regardless of fabric arrival order.**

`uneven` (N=1009 → 1024): host pads with `+inf`; squared distance stays `+inf`, loses every comparison; padded indices ≥ N anyway.

## 5. What 2× more time would buy

1. **DSD-vectorized distance compute (`@fmacs` over `mem1d_dsd`).** Scalar inner loop is `~4·d` cyc/row; DSD form is `~d` cyc/row. **Saves ~12 K cyc/PE on baseline (~75 % of total).** Pattern lifts from `tutorials/gemv-02-memory-dsds`.

2. **Threshold-pruned wavelet emission in row-gather.** After the first PE's K pairs land at `(0, py)`, a running K-th-best threshold exists; subsequent PEs emit only pairs that beat it (typically ≪ K). Reduces gather wavelets from `P·K·2` to `~K·2` on average. Needs a custom route rather than stock `mpi_x.gather`.
