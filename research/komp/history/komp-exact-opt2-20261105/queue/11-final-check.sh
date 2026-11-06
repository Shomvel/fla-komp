set -e
python validate_gpu.py --mode preflight
python validate_gpu.py --mode full
python edge_acceptance.py
python validate_scan.py
python validate_versions.py
python train_smoke.py
