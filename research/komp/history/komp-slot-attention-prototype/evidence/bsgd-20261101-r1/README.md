# BSGD merging prototype acceptance

All 402 validation checks and 8 benchmark rows completed on a dedicated H100, exit code 0. Implementation and math: ../../BSGD_MERGING.md. The unmodified tested implementation is frozen in work/ alongside the then-current block KOMP dependencies. No existing KOMP kernels or training runs were modified.

Reference: Wang et al. 2012 §6.3, using the user-linked discussion. Two adapted policies are implemented: learned-gate cyclic merging and finite-grid RKHS-degradation search among incoming-old pairs. This is not a reproduction of the SVM update or the paper's full pair/center selection.

Evidence: work/evidence/bsgd-validation.json, work/evidence/bsgd-benchmark.json, environment.json, source-sha256.json, verification.json, validation.log and benchmark.log. Job identity, terminal state and cleanup receipts are saved separately. Remote root /gpfs/users/linlizhi/bsgd-merging-20261101-r1.
