# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Rebuild KOMP result tables from the retained measurement receipts."""

import argparse
import csv
import difflib
import hashlib
import io
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ARCHIVE = ROOT / "history/20261113"
OUTPUT = ROOT / "results/20261113"
EVAL = ARCHIVE / "komp-flame-downstream-eval-20261107"
OPT = ARCHIVE / "komp-learned123-c128-opt-20261109"
MODELS = ["kda", "block", "block123", "learned", "learned123", "value", "exact123", "exact_partial"]
LABELS = {"kda": "KDA", "block": "Block", "block123": "Block+123", "learned": "Learned",
          "learned123": "Learned+123", "value": "Value", "exact123": "Exact+123", "exact_partial": "Exact partial"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def table(headers, rows):
    values = [list(map(str, headers)), *[list(map(str, row)) for row in rows]]
    widths = [max(3, *(len(row[i]) for row in values)) for i in range(len(headers))]
    lines = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in values]
    lines.insert(1, "| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(lines)


def downstream():
    source = EVAL / "FINAL_SCORES.csv"
    rows = list(csv.DictReader(io.StringIO(source.read_text())))
    verification = read(EVAL / "evidence/final-verification.json")
    assert verification["status"] == "passed"
    assert len(rows) == verification["total_items"] == 184
    assert sum(int(row["count"]) for row in rows) == verification["total_results"] == 163656
    assert len({(row["model"], row["case"]) for row in rows}) == len(rows)
    lookup = {}
    for row in rows:
        model = verification["models"][row["model"]]
        case = model["cases"][row["case"]]
        assert row["plan_sha"] == verification["plan_sha"]
        assert row["checkpoint_sha"] == model["checkpoint_sha"]
        assert int(row["tokens"]) == model["tokens"]
        assert (row["partial"] == "True") == model["partial"]
        assert int(row["count"]) == case["count"]
        assert float(row["score"]) == case["score"]
        assert 0 <= float(row["score"]) <= 1
        lookup[row["model"], row["case"]] = float(row["score"])
    summary = {}
    for model in MODELS:
        assert sum(int(row["count"]) for row in rows if row["model"] == model) == 20457
        summary[model] = {
            f"single{variant}_mean": statistics.mean(lookup[model, f"niah_single_{variant}_{length}"]
                                                     for length in [1024, 2048, 4096, 8192])
            for variant in [1, 2, 3]
        }
    lines = ["# KOMP 340M 下游分数", "", "8 个 checkpoint，共 184 个分项、163,656 条结果；每模型 23 项、20,457 条。",
             "Exact partial 训练 3,758,096,384 tokens，其余模型训练 6,800,000,000 tokens。每项分数单位为百分比。",
             "", "FP32 权重、BF16 autocast、greedy、完整前缀重算、无 cache、无 padding。",
             "NIAH 每项 500 例；PhoneBook 每档 1,000 例，历史 single_break prompt 约 80–244 tokens。",
             "BASED 沿用 cap2048 并左截断到 2000 tokens；SWDE/FDA/NQ/DROP 分别为 1111/1102/3157/2087 例。",
             "训练 context 为 2048，4K/8K 为外推。单 checkpoint 无多 seed 显著性结论。", ""]
    for variant in [1, 2, 3]:
        lines += [f"## NIAH single{variant}", "", table(
            ["模型", "1024", "2048", "4096", "8192", "四长度均值"],
            [[LABELS[model], *[f"{lookup[model, f'niah_single_{variant}_{length}'] * 100:.2f}"
                               for length in [1024, 2048, 4096, 8192]],
              f"{summary[model][f'single{variant}_mean'] * 100:.2f}"] for model in MODELS]), ""]
    lines += ["## BASED", "", table(["模型", "SWDE", "FDA", "NQ", "DROP"],
              [[LABELS[model], *[f"{lookup[model, case] * 100:.2f}" for case in
                                 ["based_swde", "based_fda", "based_nq_1024", "based_drop"]]] for model in MODELS]), "",
              "## PhoneBook", "", table(["模型", "32", "64", "100", "128", "150", "200", "256"],
                                        [[LABELS[model], *[f"{lookup[model, f'phonebook_{size}'] * 100:.2f}"
                                                           for size in [32, 64, 100, 128, 150, 200, 256]]] for model in MODELS]), "",
              "## 来源", "", "原精度分数见 [DOWNSTREAM_SCORES.csv](DOWNSTREAM_SCORES.csv)，范围与描述性均值见 [summary.json](summary.json)。",
              "固定 plan、checkpoint 和 34,000 个 batch 的历史哈希核验见 "
              "[final-verification.json](../../history/20261113/komp-flame-downstream-eval-20261107/evidence/final-verification.json)。",
              "本次重建只读取已保留的结果和验收回执，没有重新运行模型。原始 batch 文件与 checkpoint 未随仓库发布。", ""]
    return source.read_text(), "\n".join(lines), summary


def loss_rows():
    source = EVAL / "evidence/final-loss-refresh-20261107.json"
    records = [(model, record, source) for model, record in read(source).items()]
    value = ARCHIVE / "komp-value-decay-learned-123-340m-20c-20261107/reports/latest.json"
    records.append(("value_learned123", read(value)["complete"], value))
    return [dict(model=model, status=record["status"], step=record["step"],
                 supervised_tokens=record["supervised_tokens"],
                 validation_ce=record["validation"]["cross_entropy"], test_ce=record["test"]["cross_entropy"],
                 validation_ppl=record["validation"]["perplexity"], test_ppl=record["test"]["perplexity"],
                 evaluated_tokens_per_split=record["validation"]["supervised_tokens"],
                 source_sha=record["source_sha"], data_sha=record["data_sha"],
                 receipt=str(path.relative_to(ROOT)), receipt_sha256=digest(path))
            for model, record, path in records]


def timing_rows():
    rows = []

    def add(experiment, implementation, shape, dtype, metric, value, memory, memory_kind, method, path):
        rows.append(dict(experiment=experiment, implementation=implementation, shape=shape, dtype=dtype,
                         metric=metric, milliseconds=value, peak_bytes=memory, memory_kind=memory_kind,
                         method=method, receipt=str(path.relative_to(ROOT)), receipt_sha256=digest(path)))

    for experiment in ["komp-gdn-8k-20261105", "komp-exact-opt-20261105", "komp-exact-opt2-20261105"]:
        path = ARCHIVE / experiment / "evidence/comparison.json"
        receipt = read(path)
        assert receipt["status"] == "passed"
        for row in receipt["rows"]:
            shape = "x".join(str(row[key]) for key in ["B", "T", "H", "K", "V"]) + f"/M{row['M']}/C{row['compute_chunk']}"
            for metric in ["forward", "forward_backward"]:
                for method in ["eager", "cuda_graph"]:
                    add(experiment, row["operator"], shape, row["dtype"], metric, row[metric][method]["ms"],
                        round(row[metric]["extra_peak_MiB"] * 2 ** 20), "eager_allocated_delta", method, path)
    path = ARCHIVE / "komp-c128-triton-20261101/evidence/benchmark-stable.json"
    for row in read(path):
        for metric in ["forward", "forward_backward"]:
            add("block_c128", row["implementation"] + "/" + row["evict"], "x".join(map(str, row["shape"])),
                row["dtype"], metric, row[metric + "_ms"], row["incremental_peak_bytes"], "allocated_delta",
                "median_of_3_measurements", path)
    path = ARCHIVE / "komp-value-decay-native-20261106-r2/evidence/acceptance.json"
    receipt = read(path)
    assert receipt["status"] == "passed"
    for row in receipt["benchmarks"]:
        add("value_decay", row["backend"] + "/" + row["gate"],
            "x".join(map(str, row["shape"])) + f"/C{row['chunk']}", "BF16_qkv_FP32_gate_state", "forward_backward",
            row["median_ms"], row["peak_allocated_bytes"], "allocated_total", "3_warmups_7_cuda_events", path)
    for name in ["baseline-operator", "decayreduce-operator", "split-operator"]:
        path = OPT / f"evidence/{name}.json"
        row = read(path)
        add("learned123_c128", name, "16x2048x32x64/M32/C128", "BF16", "forward_backward",
            row["median_seconds"] * 1000, row["peak_bytes"], "allocated_total", "operator_warm_median", path)
    path = OPT / "evidence/final-v2-model-abba.json"
    for index, row in enumerate(read(path)["abba"]):
        add("learned123_1b_model", ("candidate" if row["candidate"] else "baseline") + f"/phase{index + 1}",
            "B16/T2048/D2048/L24/FF3072/H32/K64/M32/C128", "BF16_autocast", "full_training_step",
            row["median_seconds"] * 1000, "", "not_recorded_per_abba_phase", "ABBA_10_warm_steps_per_phase", path)
    path = ROOT / "evidence/frozen-20261106/abba-summary.json"
    for row in read(path)["runs"]:
        add("exact340m_model_cache", row["variant"] + "/" + row["run"], "B16x2/T2048/339966848_params",
            "BF16_autocast", "full_training_step", row["median_seconds"] * 1000, row["peak_allocated"],
            "allocated_total", "ABBA_10_warm_steps_per_phase", path)
    return rows


def verify_archive():
    manifest = read(ARCHIVE / "manifest.json")
    paths = set()
    for entry in manifest["files"]:
        path = REPO / entry["path"]
        canonical = REPO / entry["canonical"]
        assert path.resolve().is_relative_to(REPO.resolve()), path
        assert canonical.resolve().is_relative_to(REPO.resolve()), canonical
        assert path.stat().st_size == entry["bytes"], path
        assert digest(path) == digest(canonical) == entry["sha256"], path
        assert entry["path"] not in paths, path
        paths.add(entry["path"])
    return len(paths)


def verify_measurement_sources():
    checks = 0
    for experiment in ["komp-gdn-8k-20261105", "komp-exact-opt-20261105", "komp-exact-opt2-20261105"]:
        base = ARCHIVE / experiment
        pins = read(base / "evidence/comparison.json")["source_sha256"]
        assert "exact_komp/operator.py" in pins
        for rel, sha in pins.items():
            path = base / rel
            if path.is_file():
                assert digest(path) == sha, path
                checks += 1
    for name, field in [("final-v2-model-abba.json", "source_files"), ("final-v2-model-adam.json", "candidate_source_files")]:
        pins = read(OPT / f"evidence/{name}")[field]
        for rel, sha in pins.items():
            path = OPT / "snapshot-final-v2" / rel
            if path.is_file():
                assert digest(path) == sha, path
                checks += 1
    adam = read(OPT / "evidence/final-v2-model-adam.json")
    assert digest(OPT / "models.py") == adam["model_adapter_sha256"]
    assert digest(OPT / "snapshot-final-v2/fla/ops/komp/block_fast.py") == read(
        OPT / "evidence/final-v2-model-abba.json")["operator_sha"]
    return checks + 2


def candidate_patch():
    files = ["fla/ops/komp/block_fast.py", "fla/models/komp/modeling_komp.py", "fla/modules/rounded_swiglu.py"]
    pairs = [(OPT / "snapshot-baseline" / rel, OPT / "snapshot-final-v2" / rel, "snapshot/" + rel) for rel in files]
    pairs.append((ARCHIVE / "komp-learned123-1b-50b-20261109-mb16-direct/models.py", OPT / "models.py", "models.py"))
    parts = []
    for before, after, label in pairs:
        original = before.read_text().splitlines(keepends=True) if before.exists() else []
        parts.extend(difflib.unified_diff(original, after.read_text().splitlines(keepends=True),
                                          fromfile="a/" + label if before.exists() else "/dev/null", tofile="b/" + label))
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Validate archive hashes and compare generated tables without writing")
    args = parser.parse_args()
    archive_files = verify_archive()
    source_hash_checks = verify_measurement_sources()
    scores, report, descriptive = downstream()
    timing = timing_rows()
    remote = read(OUTPUT / "remote-filesystem.json")
    run = remote["runs"]["komp-learned123-1b-50b-20261110-resume1"]
    platform = read(OUTPUT / "remote-platform.json")
    assert len(platform) == 1 and platform[0]["name"] == "job-35a0f599ed70-20261110172046"
    summary = dict(
        archive_files=archive_files, measurement_source_hash_checks=source_hash_checks,
        downstream_models=8, downstream_cases=184, downstream_examples=163656,
        downstream_score_scale="fraction_0_to_1", descriptive_means=descriptive, kernel_timing_rows=len(timing),
        learned123_single_h100=dict(
            baseline_seconds=statistics.mean(row["median_seconds"] for row in read(OPT / "evidence/final-v2-model-abba.json")["abba"]
                                             if not row["candidate"]),
            candidate_seconds=statistics.mean(row["median_seconds"] for row in read(OPT / "evidence/final-v2-model-abba.json")["abba"]
                                              if row["candidate"])),
        learned123_1b=dict(captured_utc=remote["captured_utc"], platform_status=platform[0]["status"],
                           latest_metrics=run["metrics"]["last"], latest_checkpoint=run["files"]["runs/block/latest.json"],
                           latest_validation=run["validations"][-1], final_50b_complete="runs/block/complete.json" in run["files"],
                           downstream_cases=0, watcher_failure=remote["runs"]["komp-learned123-1b-downstream-eval-20261110"]
                           ["files"]["watcher/failure.json"]),
    )
    summary["learned123_single_h100"]["speedup"] = (
        summary["learned123_single_h100"]["baseline_seconds"] / summary["learned123_single_h100"]["candidate_seconds"]
    )
    outputs = {"DOWNSTREAM_SCORES.csv": scores, "MODEL_RESULTS.md": report, "MODEL_LOSS.csv": csv_text(loss_rows()),
               "KERNEL_TIMINGS.csv": csv_text(timing), "../../history/20261113/learned123-v2.patch": candidate_patch(),
               "summary.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n"}
    for name, content in outputs.items():
        path = OUTPUT / name
        if args.check:
            assert path.read_text() == content, f"Stale generated result: {path}"
        else:
            path.write_text(content)
    print(json.dumps(dict(status="passed", archive_files=archive_files, downstream_cases=184,
                          downstream_examples=163656, kernel_timing_rows=len(timing), mode="check" if args.check else "write")))


if __name__ == "__main__":
    main()
