set -euo pipefail
python3 probe.py --mode operator --snapshot snapshot-ut-separate --tag ut-separate4 --profile --flags '{"UT_SOLVE_WARPS":4}' --checkpoint fixture/checkpoint-010240
