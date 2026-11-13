set -euo pipefail
python3 sweep.py --mode operator --snapshot snapshot --tag ut-warps --warmup 2 --reps 6 --checkpoint fixture/checkpoint-010240
