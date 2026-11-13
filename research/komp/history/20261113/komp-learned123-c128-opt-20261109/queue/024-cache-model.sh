set -euo pipefail
python3 probe.py --mode model --snapshot snapshot-cache --tag cache-model --profile --cache-ut-layers 24 --checkpoint fixture/checkpoint-010240
