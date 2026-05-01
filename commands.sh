#!/usr/bin/env bash
# Compile + run all 6 grader cases against v4 (hybrid). Same parameter table
# as tests/test_correctness.py but invokes layout_v4.csl / pe_program_v4.csl
# / run_v4.py instead of the v1 set.
#
# Usage:  bash run_all_cases_v4.sh
set -uo pipefail

CASES=(
  "baseline   P:4,d_dim:32,rows_per_pe:128,K:16"
  "k_eq_1     P:2,d_dim:32,rows_per_pe:256,K:1"
  "k_large    P:2,d_dim:16,rows_per_pe:256,K:256"
  "uneven     P:4,d_dim:32,rows_per_pe:64,K:16"
  "all_equal  P:2,d_dim:16,rows_per_pe:256,K:16"
  "duplicates P:2,d_dim:16,rows_per_pe:256,K:8"
)

PASS=0
FAIL=0
RESULTS=()

mkdir -p results_v4

for spec in "${CASES[@]}"; do
  name=$(echo "$spec" | awk '{print $1}')
  params=$(echo "$spec" | awk '{print $2}')
  build_dir="out_v4_$name"

  echo
  echo "═══════════════════════════════════════════════════════════"
  echo " [v4] Case: $name   (params: $params)"
  echo "═══════════════════════════════════════════════════════════"

  rm -rf "$build_dir"

  if ! cslc --arch=wse2 layout.csl \
        --fabric-dims=11,6 --fabric-offsets=4,1 \
        --params="$params" \
        --memcpy --channels=1 \
        -o "$build_dir"; then
    echo "✗ FAIL ($name): compile error"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL  $name  (compile)")
    continue
  fi

  log="results_v4/${name}.log"
  mkdir -p "$(dirname "$log")"
  if cs_python run.py --name "$build_dir" --case "$name" 2>&1 \
       | tee "$log" \
       | grep -q "^PASS: $name"; then
    echo "✓ PASS ($name)"
    PASS=$((PASS + 1))
    RESULTS+=("PASS  $name")
  else
    echo "✗ FAIL ($name): run error or wrong output (see $log)"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL  $name  (run)")
  fi
done

echo
echo "═══════════════════════════════════════════════════════════"
echo " [v4] Summary: ${PASS} passed, ${FAIL} failed (of ${#CASES[@]})"
echo "═══════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo

[ $FAIL -eq 0 ]
