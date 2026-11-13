set -euo pipefail
python3 probe.py --mode model --snapshot snapshot --tag decayreduce-model --profile --checkpoint fixture/checkpoint-010240 --warmup 3 --reps 8
