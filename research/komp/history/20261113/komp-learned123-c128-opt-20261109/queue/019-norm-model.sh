set -euo pipefail
python3 probe.py --mode model --snapshot snapshot-norm --tag norm-model --profile --checkpoint fixture/checkpoint-010240
