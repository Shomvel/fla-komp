set -e
python selection_check.py
python validate_gpu.py --mode preflight
python validate_gpu.py --mode full
python edge_acceptance.py
python train_smoke.py
