set -e
python validate_gpu.py --mode preflight
python edge_acceptance.py
