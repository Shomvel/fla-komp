set -euo pipefail
python3 probe.py --mode operator --snapshot snapshot-ut --tag ut-operator --profile --checkpoint fixture/checkpoint-010240
