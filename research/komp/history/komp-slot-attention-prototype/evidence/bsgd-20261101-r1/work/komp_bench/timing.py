"""CUDA Graph replay timing with a bounded unroll, so capture never grows the pool unboundedly."""
import statistics
import torch


def graph_ms(fn, unroll=10, replays=20, trials=3, mode='global'):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode=mode):
        for _ in range(unroll):
            fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end)/(replays*unroll))
    del g
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return statistics.median(out)
