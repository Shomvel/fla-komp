set -e
cp evidence/comparison.json evidence/candidate1-comparison.json
cp evidence/kernel-profile.json evidence/candidate1-kernel-profile.json
export FLA_DISABLE_BACKEND_DISPATCH=1 FLA_DISABLE_TENSOR_CACHE=1
python compare.py
python profile_kernels.py
