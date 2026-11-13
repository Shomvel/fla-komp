set -euo pipefail
python3 probe.py --mode model --checkpoint /gpfs/users/linlizhi/komp-learned123-c128-opt-20261109/fixture/checkpoint-010240 --snapshot snapshot-baseline --tag baseline-model --profile --warmup 3 --reps 8
