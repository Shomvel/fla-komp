set -euo pipefail
python3 probe.py --mode model --snapshot snapshot-ff --tag ff-model --profile --checkpoint fixture/checkpoint-010240
