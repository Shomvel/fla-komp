set -e
export FLA_DISABLE_BACKEND_DISPATCH=1 FLA_DISABLE_TENSOR_CACHE=1
python version_counts.py
