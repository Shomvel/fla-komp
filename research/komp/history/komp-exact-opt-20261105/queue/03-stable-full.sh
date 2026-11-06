set -e
cp evidence/full-failure.json evidence/fp32-naive-full-failure.json
python validate_gpu.py --mode full
