#!/usr/bin/env cs_python
"""Host driver for the top-K kNN kernel — hybrid v4.

Differences vs run.py (v1):
  • q replicated to every PE via H2D (no on-fabric q broadcast).
  • Padding sentinel is +inf (intent-clear, dim-independent).
  • Reads results from PE (0, 0) instead of (P-1, P-1).
  • Symbol names: D_shard / query / topk_dists / topk_idxs / kernel.

Invoked by the grader as:
    cs_python run.py --name <build_dir> --case <case_name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

# Grader brings its own reference.py at the same level as run.py.
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
    parser.add_argument("--cmaddr",
                        help="IP:port for a CS hardware system; omit for sim")
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

    # Build (D, q) and sanity-check it matches the compile params.
    case = CASE_BUILDERS[args.case]()
    D, q, K_case, P_case = case["D"], case["q"], case["K"], case["P"]
    assert P_case == P, f"P mismatch: case={P_case}, compiled={P}"
    assert K_case == K, f"K mismatch: case={K_case}, compiled={K}"
    assert D.shape[1] == d_dim, f"d mismatch: case={D.shape[1]}, compiled={d_dim}"
    assert D.dtype == np.float32 and q.dtype == np.float32

    N = D.shape[0]
    N_padded = P * P * R
    assert N <= N_padded, f"N={N} larger than grid capacity {N_padded}"

    # Pad with +inf rows: padded distances are +inf (squared diff overflows
    # to +inf and stays +inf), so they can never enter any PE's top-K.
    if N < N_padded:
        pad_rows = N_padded - N
        pad = np.full((pad_rows, d_dim), np.float32("inf"), dtype=np.float32)
        D_padded = np.concatenate([D, pad], axis=0)
    else:
        D_padded = D

    # Oracle (built from the unpadded D — padded indices must never appear).
    idx_expected, dist_expected = topk_reference(D, q, K)

    # Reshape D for ROW_MAJOR memcpy_h2d. Hardware mapping:
    #   PE (px, py) gets rows [(py*P + px)*R, (py*P + px + 1)*R).
    # MemcpyOrder.ROW_MAJOR with (h=P, w=P, l=R*d) writes the flat host array
    # so that index [py, px, i*d+j] = D_padded[(py*P + px)*R + i, j].
    D_for_memcpy = (D_padded
                    .reshape(P, P, R, d_dim)
                    .reshape(P, P, R * d_dim))

    # Replicate q to every PE: avoids the on-fabric broadcast entirely.
    q_replicated = np.tile(q, P * P).reshape(P, P, d_dim)

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

    sym_D    = runner.get_id("D_shard")
    sym_q    = runner.get_id("query")
    sym_outd = runner.get_id("topk_dists")
    sym_outi = runner.get_id("topk_idxs")

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

    # 2. q replicated to every PE.
    runner.memcpy_h2d(
        sym_q, q_replicated.ravel(),
        0, 0, P, P, d_dim,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )

    # 3. Run the kernel.
    runner.launch("kernel", nonblock=False)

    # 4. Read K (dist, idx) pairs from PE (0, 0).
    out_dist    = np.zeros(K, dtype=np.float32)
    out_idx_u32 = np.zeros(K, dtype=np.uint32)
    runner.memcpy_d2h(
        out_dist, sym_outd,
        0, 0, 1, 1, K,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    runner.memcpy_d2h(
        out_idx_u32, sym_outi,
        0, 0, 1, 1, K,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )

    runner.stop()

    out_idx = out_idx_u32.astype(np.int32)

    # 5. Compare. Indices must match exactly (lex tie-break by smaller idx);
    #    distances must match within atol/rtol = 1e-3 per SPEC.
    passed = True
    fail_msg = ""
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
        passed = False
        fail_msg = str(e)

    print(f"\n[case={args.case}] top-{K} from device vs oracle:")
    print(f"  device idx:   {out_idx.tolist()}")
    print(f"  oracle idx:   {idx_expected.tolist()}")
    print(f"  device dist:  {[round(float(x), 6) for x in out_dist]}")
    print(f"  oracle dist:  {[round(float(x), 6) for x in dist_expected]}")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"{args.case}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case":        args.case,
                "passed":      passed,
                "params":      cp,
                "device_idx":  out_idx.tolist(),
                "oracle_idx":  idx_expected.tolist(),
                "device_dist": [float(x) for x in out_dist],
                "oracle_dist": [float(x) for x in dist_expected],
            },
            f,
            indent=2,
        )
    print(f"  saved → {result_path.relative_to(Path.cwd())}")

    if not passed:
        print(f"FAIL: {args.case}", file=sys.stderr)
        print(fail_msg, file=sys.stderr)
        return 1

    print(f"PASS: {args.case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
