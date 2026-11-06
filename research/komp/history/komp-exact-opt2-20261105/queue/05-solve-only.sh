set -e
python validate_gpu.py --mode full
python edge_acceptance.py
