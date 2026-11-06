set -e
python integration_check.py
python edge_acceptance.py
export FLA_DISABLE_BACKEND_DISPATCH=1 FLA_DISABLE_TENSOR_CACHE=1
python compare.py
