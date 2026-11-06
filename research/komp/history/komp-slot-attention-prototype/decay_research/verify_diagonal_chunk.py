"""Independent FP64 check of diagonal-decay rank-two chunk transfer maps.

This checks the algebra using NumPy, not the production implementation or GPU.
Run from any directory with NumPy installed; JSON is saved beside this script.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(731)
    C, M, V = 5, 4, 3
    cases = []
    for zero in (False, True):
        a = rng.uniform(.1, 1, (C, M))
        beta = rng.uniform(.1, .7, C)
        if zero:
            a[1, 2] = 0
            a[3, :] = 0
        e = np.eye(M)[[0, 0, 2, 1, 3]]
        p = rng.normal(0, .2, (C, M))
        h = rng.normal(0, .2, (C, M))
        r = rng.normal(size=(C, M))
        v = rng.normal(size=(C, V))
        s0 = rng.normal(size=(M, V))
        X = np.stack((p-e, -beta[:, None]*e), axis=-1)
        Y = np.stack((e, h), axis=-1)
        w = beta[:, None]*e

        # Inclusive product; an empty time interval is the identity diagonal.
        def P(i, j):
            return np.prod(a[j:i+1], axis=0) if i >= j else np.ones(M)

        L = np.zeros((2*C, 2*C))
        Y0 = np.zeros((2*C, M))
        B = np.zeros((2*C, C))
        RX = np.zeros((C, 2*C))
        RW = np.zeros((C, C))
        R0 = np.zeros((C, M))
        EX = np.zeros((M, 2*C))
        EW = np.zeros((M, C))
        for i in range(C):
            Y0[2*i:2*i+2] = Y[i].T*P(i, 0)
            R0[i] = r[i]*P(i, 0)
            for j in range(i):
                L[2*i:2*i+2, 2*j:2*j+2] = (Y[i].T*P(i, j+1))@X[j]
                B[2*i:2*i+2, j] = (Y[i].T*P(i, j+1))@w[j]
            for j in range(i+1):
                RX[i, 2*j:2*j+2] = (r[i]*P(i, j+1))@X[j]
                RW[i, j] = (r[i]*P(i, j+1))@w[j]
        for j in range(C):
            EX[:, 2*j:2*j+2] = P(C-1, j+1)[:, None]*X[j]
            EW[:, j] = P(C-1, j+1)*w[j]
        Z0 = np.linalg.solve(np.eye(2*C)-L, Y0)
        Zv = np.linalg.solve(np.eye(2*C)-L, B)
        OS = R0+RX@Z0
        OV = RW+RX@Zv
        FS = np.diag(P(C-1, 0))+EX@Z0
        FV = EW+EX@Zv

        outputs = []
        state = s0.copy()
        for i in range(C):
            pre = a[i, :, None]*state
            state = pre+X[i]@(Y[i].T@pre)+w[i, :, None]*v[i, None, :]
            outputs.append(r[i]@state)
        output_error = float(np.max(abs(OS@s0+OV@v-np.array(outputs))))
        state_error = float(np.max(abs(FS@s0+FV@v-state)))
        err = max(output_error, state_error)
        assert err < 1e-12, err
        cases.append({"zero_gate": zero, "output_max_abs": output_error,
                      "final_state_max_abs": state_error, "max_abs": err,
                      "tolerance": 1e-12, "passed": True})
        print('diagonal rank2 chunk, zero_gate=', zero, 'max error=', err)

    path = Path(__file__).resolve()
    report = {"scope": "Independent NumPy FP64 algebra check; no production/GPU validation",
              "seed": 731, "C": C, "M": M, "V": V,
              "slots": [0, 0, 2, 1, 3], "numpy_version": np.__version__,
              "script_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "cases": cases, "passed": True}
    path.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
