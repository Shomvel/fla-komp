set -euo pipefail
python validate_c128.py
python validate_block_komp.py --output ../evidence/legacy-validation.json
