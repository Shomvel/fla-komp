set -euo pipefail
python3 probe.py --mode model --snapshot snapshot-final-v2 --tag final-v2-model-abba --cache-ut-layers 24 --compare-reference snapshot-baseline --profile --checkpoint fixture/checkpoint-010240
