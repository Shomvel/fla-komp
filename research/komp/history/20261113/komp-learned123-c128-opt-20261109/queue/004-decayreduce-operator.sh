set -euo pipefail
python3 probe.py --mode operator --snapshot snapshot --tag decayreduce-operator --profile --checkpoint fixture/checkpoint-010240 --warmup 4 --reps 10
