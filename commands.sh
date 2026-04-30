#!/usr/bin/env bash
# Compile + run the baseline test case. Mirrors what tests/test_correctness.py does
# for the 'baseline' case (P=4, d=32, rows_per_pe=128, K=16).
set -euo pipefail

cslc --arch=wse2 layout.csl \
  --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 \
  --memcpy --channels=1 \
  -o out

cs_python run.py --name out --case baseline
