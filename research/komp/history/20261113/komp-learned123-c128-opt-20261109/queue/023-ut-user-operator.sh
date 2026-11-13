set -euo pipefail
python3 probe.py --mode operator --snapshot snapshot-ut --tag ut-user-accepted-operator --profile --checkpoint fixture/checkpoint-010240
python3 export_metadata.py
