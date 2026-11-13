set -euo pipefail
python3 validate_variant.py --snapshot snapshot-ut --tag ut-user-accepted-differential --quick --rms-tol 0.001
