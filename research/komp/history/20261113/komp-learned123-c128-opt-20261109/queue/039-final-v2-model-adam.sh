set -euo pipefail
python3 validate_model_variant.py --snapshot snapshot-final-v2 --tag final-v2-model-adam --cache-ut-layers 24 --loss-atol .001 --optimizer-check
