"""Independent prefix, adjoint, streaming and precision checks on frozen Block-KOMP."""
import json
from pathlib import Path
import torch
from validate_block_komp import inputs, run, rms
from komp_slot.block_fast import block_komp_fast as candidate
from komp_slot.block_reference import block_komp_reference as reference

DEST = Path('evidence/extended.json')
report = {'checks': []}


def record(row):
    report['checks'].append(row)
    DEST.write_text(json.dumps(report, indent=2))
    assert row['passed'], row


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    for dtype in (torch.float32, torch.bfloat16):
        for c in (16, 32, 64):
            # More than sixteen chunks exercises both levels of the affine scan.
            xs = inputs(1, 17*c+3, 2, 32, 24, 16, dtype, 20261101+c)
            with torch.no_grad():
                full = run(candidate, xs, c)
                cuts = sorted({1, c//2, c-1, c, c+1, 2*c-1, 4*c+3, 8*c, xs[0].shape[1]-1})
                for cut in cuts:
                    for component in range(5):
                        altered = [x.clone() for x in xs]
                        tail = altered[component][:, cut:]
                        if component == 3:
                            tail.copy_(-torch.rand_like(tail)*2)
                        elif component == 4:
                            tail.copy_(torch.rand_like(tail))
                        else:
                            tail.copy_(torch.randn_like(tail)*3)
                        got = run(candidate, altered, c)[0][:, :cut]
                        err = float((got.float()-full[0][:, :cut].float()).abs().max())
                        record(dict(test='suffix_intervention', dtype=str(dtype), c=c, cut=cut,
                                    component=component, max_abs=err, passed=err == 0))
                cut = 8*c
                first = candidate(*[x[:, :cut] for x in xs[:5]], (xs[5], xs[6]), chunk_size=c)
                second = candidate(*[x[:, cut:] for x in xs[:5]], first[1], chunk_size=c)
                errors = [rms(torch.cat((first[0], second[0]), 1), full[0]),
                          rms(second[1][0], full[1]), rms(second[1][1], full[2])]
                record(dict(test='aligned_stream', dtype=str(dtype), c=c, errors=errors,
                            tolerance=.01, passed=max(errors) < .01))
            # Prefix-only loss must have zero derivative w.r.t. all future inputs,
            # including the keys used to construct this chunk's final dictionary.
            for cut in (c//2, c, c+1, 4*c+3):
                leaves = [x.detach().clone().requires_grad_() for x in xs]
                out = run(candidate, leaves, c)[0]
                grad = torch.autograd.grad(out[:, :cut].float().square().mean(), leaves)
                future = [float(x[:, cut:].abs().max()) for x in grad[:5]]
                record(dict(test='future_gradient', dtype=str(dtype), c=c, cut=cut,
                            max_abs=future, tolerance=1e-7, passed=max(future) <= 1e-7))
            print('PASS causality', dtype, c, flush=True)

    # Reference gradients under separate output-only and final-state-only losses.
    # Exact beta=0/1, no decay and near-complete reset exercise cancellation paths.
    for dtype in (torch.float32, torch.bfloat16):
        for c, m in ((16, 32), (32, 32), (64, 16)):
            xs = inputs(1, 3*c+5, 1, 32, 17, m, dtype, 317+c)
            xs[3].zero_()
            xs[3][:, c+1] = -100
            xs[4][:, ::3] = 0
            xs[4][:, 1::3] = 1
            for loss in ('output', 'state'):
                xx = [x.detach().clone().requires_grad_() for x in xs]
                rr = [x.detach().double().requires_grad_() for x in xs]
                got, ref = run(candidate, xx, c), run(reference, rr, c)
                weights = [torch.randn_like(x) for x in got]
                if loss == 'output':
                    weights[1].zero_(); weights[2].zero_()
                else:
                    weights[0].zero_()
                gg = torch.autograd.grad(got, xx, weights)
                gr = torch.autograd.grad(ref, rr, [w.double() for w in weights])
                tol = .005 if dtype == torch.float32 else .01
                errors = {}
                passed = True
                for name, a, b in zip(('out', 'keys', 'values', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'dv0'), (*got, *gg), (*ref, *gr)):
                    abs_error = float((a.double()-b).abs().max())
                    rel = rms(a, b)
                    errors[name] = dict(rms_relative=rel, max_abs=abs_error)
                    passed &= bool(torch.isfinite(a).all()) and (rel <= tol or abs_error <= 1e-7)
                record(dict(test='edge_reference', dtype=str(dtype), c=c, m=m, loss=loss,
                            tolerance=tol, errors=errors, passed=passed))
                print('PASS edge reference', dtype, c, loss, flush=True)


if __name__ == '__main__':
    try:
        main()
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        DEST.write_text(json.dumps(report, indent=2))
