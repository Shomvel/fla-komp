set -euo pipefail
python3 probe.py --mode model --snapshot snapshot --tag abba-model --warmup 2 --reps 8 --abba --checkpoint fixture/checkpoint-010240
