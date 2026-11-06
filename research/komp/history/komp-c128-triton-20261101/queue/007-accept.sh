set -euo pipefail
python resources_c128.py
python validate_c128.py --smoke --output smoke-final.json
python validate_c128.py
python validate_block_komp.py --output ../evidence/legacy-validation.json
