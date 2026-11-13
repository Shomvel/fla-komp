set -euo pipefail
python3 validate_model_variant.py --snapshot snapshot-final --tag final-model-adam --cache-ut-layers 24 --loss-atol .001 --optimizer-check
