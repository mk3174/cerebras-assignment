#!/usr/bin/env cs_python
"""Host driver for the top-K kNN kernel.

Invoked by the grader as:
    cs_python run.py --name <build_dir> --case <case_name>

Steps:
  1. Read the compile params from <build_dir>/out.json.
  2. Look up the corresponding test case from reference.py.
  3. Pad D out to P*P*rows_per_pe rows with +inf rows so distances on padded
     rows are infinite (and never selected). Padded indices fall outside [0, N)
     and would lose any tie-break anyway.
  4. memcpy_h2d the per-PE D shards (ROW_MAJOR, l = rows_per_pe * d_dim per PE).
  5. memcpy_h2d the full query q into PE (0, 0) only.
  6. runner.launch("kernel"), which runs the state machine on every PE.
  7. memcpy_d2h the K (dist, idx) pairs from PE (P-1, P-1).
  8. Compare against the NumPy oracle (reference.py:topk_reference) and
     print 'PASS: <case>' on success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

# reference.py lives at the repo root, alongside this run.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import (  # noqa: E402  pylint: disable=wrong-import-position
    make_all_equal,
    make_baseline,
    make_duplicates,
    make_k_eq_1,
    make_k_large,
    make_uneven,
    topk_reference,
)


# Map grader's --case names to the case-builder fns from reference.py.
CASE_BUILDERS = {
    "baseline":   make_baseline,
    "k_eq_1":     make_k_eq_1,
    "k_large":    make_k_large,
    "uneven":     make_uneven,
    "all_equal":  make_all_equal,
    "duplicates": make_duplicates,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="path to the cslc build dir (e.g. ./out)")
    parser.add_argument("--case", required=True, choices=list(CASE_BUILDERS),
                        help="which test case to run")
    parser.add_argument("--cmaddr", help="IP:port for a CS hardware system; omit for sim")
    return parser.parse_args()


def load_compile_params(build_dir: str) -> dict[str, int]:
    out_json_path = Path(build_dir) / "out.json"
    with out_json_path.open(encoding="utf-8") as f:
        compile_data = json.load(f)
    params = compile_data["params"]
    return {
        "P":           int(params["P"]),
        "d_dim":       int(params["d_dim"]),
        "rows_per_pe": int(params["rows_per_pe"]),
        "K":           int(params["K"]),
    }


def main() -> int:
    args = parse_args()
    cp = load_compile_params(args.name)
    P, d_dim, R, K = cp["P"], cp["d_dim"], cp["rows_per_pe"], cp["K"]

    # Build the test case (D, q) and sanity-check it matches the compile params.
    case = CASE_BUILDERS[args.case]()
    D, q, K_case, P_case = case["D"], case["q"], case["K"], case["P"]
    assert P_case == P, f"P mismatch: case={P_case}, compiled={P}"
    assert K_case == K, f"K mismatch: case={K_case}, compiled={K}"
    assert D.shape[1] == d_dim, f"d mismatch: case={D.shape[1]}, compiled={d_dim}"
    assert D.dtype == np.float32 and q.dtype == np.float32

    N = D.shape[0]
    N_padded = P * P * R
    assert N <= N_padded, f"N={N} larger than grid capacity {N_padded}"

    # Pad D with +inf rows so padded distances are inf (never selected).
    if N < N_padded:
        pad_rows = N_padded - N
        pad = np.full((pad_rows, d_dim), np.float32("inf"), dtype=np.float32)
        D_padded = np.concatenate([D, pad], axis=0)
    else:
        D_padded = D

    # Oracle (built from the unpadded D — padded rows must never appear in output).
    idx_expected, dist_expected = topk_reference(D, q, K)

    # Reshape D for the ROW_MAJOR memcpy_h2d. Layout requirement:
    #   PE (px, py) gets rows [(py*P + px)*R : (py*P + px + 1)*R].
    # MemcpyOrder.ROW_MAJOR maps a flat host array as h × w × l with l fastest, where
    # l = rows_per_pe * d_dim, w = P (cols), h = P (rows). The flattening of
    # D_padded.reshape(P, P, R, d) makes index [py, px, i, j] = D_padded[(py*P + px)*R + i, j],
    # which is exactly the ordering the hardware expects.
    D_for_memcpy = D_padded.reshape(P, P, R, d_dim).reshape(P, P, R * d_dim)

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

    sym_D    = runner.get_id("D_tile")
    sym_q    = runner.get_id("q")
    sym_outd = runner.get_id("out_dist")
    sym_outi = runner.get_id("out_idx")

    runner.load()
    runner.run()

    # 1. D shards to all PEs.
    runner.memcpy_h2d(
        sym_D, D_for_memcpy.ravel(),
        0, 0, P, P, R * d_dim,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=True,
    )

    # 2. q to PE (0, 0) only.
    runner.memcpy_h2d(
        sym_q, q,
        0, 0, 1, 1, d_dim,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )

    # 3. Run the kernel.
    runner.launch("kernel", nonblock=False)

    # 4. Read back K distances and indices from PE (P-1, P-1).
    out_dist = np.zeros(K, dtype=np.float32)
    out_idx_u32 = np.zeros(K, dtype=np.uint32)
    runner.memcpy_d2h(
        out_dist, sym_outd,
        P - 1, P - 1, 1, 1, K,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    runner.memcpy_d2h(
        out_idx_u32, sym_outi,
        P - 1, P - 1, 1, 1, K,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )

    runner.stop()

    out_idx = out_idx_u32.astype(np.int32)

    # 5. Compare. Indices must match exactly (lex tie-break by smaller idx);
    #    distances must match within atol/rtol = 1e-3 per SPEC.
    try:
        np.testing.assert_array_equal(
            out_idx, idx_expected,
            err_msg=(
                f"indices mismatch on case={args.case}\n"
                f"  got      = {out_idx.tolist()}\n"
                f"  expected = {idx_expected.tolist()}"
            ),
        )
        np.testing.assert_allclose(
            out_dist, dist_expected, atol=1e-3, rtol=1e-3,
            err_msg=(
                f"distances mismatch on case={args.case}\n"
                f"  got      = {out_dist.tolist()}\n"
                f"  expected = {dist_expected.tolist()}"
            ),
        )
    except AssertionError as e:
        print(f"FAIL: {args.case}", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1

    print(f"PASS: {args.case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
