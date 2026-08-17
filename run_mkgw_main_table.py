import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys

from scipy.stats import ttest_rel


DATASET = "MKG-W"
METHODS = ("DHNS", "MMA-HNS")
METRICS = (
    ("mrr", "MRR"),
    ("hit1", "H@1"),
    ("hit3", "H@3"),
    ("hit10", "H@10"),
)
MISSING_TEXT_SPLIT = "head_or_tail_missing_text"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the MKG-W 10/30/50% missing-text main-table experiment for "
            "DHNS vs full MMA-HNS and perform paired t-tests on seed-level overall MRR."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--train-times", type=int, default=None)
    parser.add_argument("--result-root", type=str, default="./results/mkgw_main_table")
    parser.add_argument("--checkpoint-root", type=str, default="./checkpoint/mkgw_main_table")
    parser.add_argument("--mask-root", type=str, default="./masks/mkgw_main_table")
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=512)
    parser.add_argument("--retrieval-mix-weight", type=float, default=0.25)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not launch experiments; rebuild the table from completed logs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed steps instead of resuming from logs containing RESULT_JSON.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Pass --no-gpu to training and HPSAC evaluation commands.",
    )
    return parser.parse_args()


def validate_args(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if len(args.seeds) < 2:
        raise ValueError("A paired t-test requires at least two paired seeds.")
    if len(set(args.rates)) != len(args.rates):
        raise ValueError("--rates must not contain duplicates.")
    for rate in args.rates:
        if not 0.0 < rate <= 1.0:
            raise ValueError("Every missing rate must be in (0, 1].")
    if args.train_times is not None and args.train_times <= 0:
        raise ValueError("--train-times must be positive.")


def rate_tag(rate):
    return "inject%d" % int(round(rate * 100))


def display_command(command):
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def load_result_json_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    payload = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if line.startswith("RESULT_JSON: "):
                payload = json.loads(line[len("RESULT_JSON: ") :])
    return payload


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write("\n")


def run_or_resume(label, command, log_path, raw_json_path, required_paths, force):
    existing_payload = None if force else load_result_json_from_log(log_path)
    required_paths_exist = all(os.path.exists(path) for path in required_paths)
    if existing_payload is not None and required_paths_exist:
        print("\n===== Resume completed: %s =====" % label, flush=True)
        print("Log: %s" % os.path.abspath(log_path), flush=True)
        write_json(raw_json_path, existing_payload)
        return existing_payload

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print("\n===== %s =====" % label, flush=True)
    print("Running: %s" % display_command(command), flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    payload = load_result_json_from_log(log_path)
    if payload is None:
        raise RuntimeError("No RESULT_JSON found after completing: %s" % label)
    missing_outputs = [path for path in required_paths if not os.path.exists(path)]
    if missing_outputs:
        raise RuntimeError("Expected output was not created: %s" % ", ".join(missing_outputs))
    write_json(raw_json_path, payload)
    return payload


def common_train_command(args, rate, seed, mask_path, checkpoint_path):
    command = [
        args.python_exe,
        "train_dhns_rotate.py",
        "--dataset",
        DATASET,
        "--seed",
        str(seed),
        "--inject-text-missing-rate",
        str(rate),
        "--text-missing-mask-strategy",
        "random",
        "--subset-eval",
        "--checkpoint-path",
        checkpoint_path,
    ]
    if args.train_times is not None:
        command.extend(["--train-times", str(args.train_times)])
    if args.no_gpu:
        command.append("--no-gpu")
    return command


def run_rate_seed(args, rate, seed):
    tag = rate_tag(rate)
    result_dir = os.path.join(args.result_root, tag)
    checkpoint_dir = os.path.join(args.checkpoint_root, tag)
    raw_dir = os.path.join(result_dir, "raw_result_json")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(args.mask_root, exist_ok=True)

    mask_path = os.path.join(args.mask_root, "%s_random_text%d_seed%d.pt" % (DATASET, int(round(rate * 100)), seed))
    dhns_checkpoint = os.path.join(checkpoint_dir, "dhns_seed%d.ckpt" % seed)
    a_checkpoint = os.path.join(checkpoint_dir, "mma_hns_a_seed%d.ckpt" % seed)
    ab_checkpoint = os.path.join(checkpoint_dir, "mma_hns_ab_seed%d.ckpt" % seed)

    dhns_command = common_train_command(args, rate, seed, mask_path, dhns_checkpoint)
    dhns_command.extend(["--save-text-missing-mask-path", mask_path])
    dhns_payload = run_or_resume(
        "%s DHNS | %s | seed=%d" % (DATASET, tag, seed),
        dhns_command,
        os.path.join(result_dir, "dhns_seed%d.log" % seed),
        os.path.join(raw_dir, "dhns_seed%d.json" % seed),
        [mask_path, dhns_checkpoint],
        args.force,
    )

    a_command = common_train_command(args, rate, seed, mask_path, a_checkpoint)
    a_command.extend(["--text-missing-mask-path", mask_path, "--use-soft-missing-text"])
    a_payload = run_or_resume(
        "%s MMA-HNS A | %s | seed=%d" % (DATASET, tag, seed),
        a_command,
        os.path.join(result_dir, "mma_hns_a_seed%d.log" % seed),
        os.path.join(raw_dir, "mma_hns_a_seed%d.json" % seed),
        [a_checkpoint],
        args.force,
    )

    ab_command = common_train_command(args, rate, seed, mask_path, ab_checkpoint)
    ab_command.extend(
        [
            "--text-missing-mask-path",
            mask_path,
            "--use-soft-missing-text",
            "--use-retrieval-missing-text",
            "--retrieval-topk",
            str(args.retrieval_topk),
            "--retrieval-pool-size",
            str(args.retrieval_pool_size),
            "--retrieval-mix-weight",
            str(args.retrieval_mix_weight),
        ]
    )
    ab_payload = run_or_resume(
        "%s MMA-HNS A+B | %s | seed=%d" % (DATASET, tag, seed),
        ab_command,
        os.path.join(result_dir, "mma_hns_ab_seed%d.log" % seed),
        os.path.join(raw_dir, "mma_hns_ab_seed%d.json" % seed),
        [ab_checkpoint],
        args.force,
    )

    mma_command = [
        args.python_exe,
        "eval_hpsac.py",
        "--dataset",
        DATASET,
        "--checkpoint-a",
        a_checkpoint,
        "--checkpoint-b",
        ab_checkpoint,
        "--inject-text-missing-rate",
        str(rate),
        "--text-missing-mask-strategy",
        "random",
        "--text-missing-mask-path",
        mask_path,
        "--retrieval-topk",
        str(args.retrieval_topk),
        "--retrieval-pool-size",
        str(args.retrieval_pool_size),
        "--lambda-grid",
        "0.10,0.20,0.25,0.30,0.40",
        "--alpha-grid",
        "0.0,0.1,0.2,0.3",
        "--min-group-queries",
        "30",
        "--safe-delta",
        "0.0002",
        "--lock-missing-text",
        "--subset-eval",
    ]
    if args.no_gpu:
        mma_command.append("--no-gpu")
    mma_payload = run_or_resume(
        "%s MMA-HNS A+B+C | %s | seed=%d" % (DATASET, tag, seed),
        mma_command,
        os.path.join(result_dir, "mma_hns_seed%d.log" % seed),
        os.path.join(raw_dir, "mma_hns_seed%d.json" % seed),
        [],
        args.force,
    )

    verify_shared_mask(rate, seed, [dhns_payload, a_payload, ab_payload, mma_payload])
    return {
        "DHNS": normalize_result("DHNS", rate, seed, dhns_payload, os.path.join(result_dir, "dhns_seed%d.log" % seed)),
        "MMA-HNS": normalize_result(
            "MMA-HNS", rate, seed, mma_payload, os.path.join(result_dir, "mma_hns_seed%d.log" % seed)
        ),
    }


def injection_checksum(payload):
    injection_info = payload.get("injection_info") or {}
    return injection_info.get("mask_checksum_sha256")


def verify_shared_mask(rate, seed, payloads):
    checksums = [injection_checksum(payload) for payload in payloads]
    if any(checksum is None for checksum in checksums):
        raise RuntimeError("Missing mask checksum for rate=%s seed=%s." % (rate, seed))
    if len(set(checksums)) != 1:
        raise RuntimeError("Methods did not use the same missing-text mask for rate=%s seed=%s." % (rate, seed))


def metric_sources(method, payload):
    if method == "DHNS":
        return payload.get("overall_metrics"), payload.get("subset_metrics") or {}
    return payload.get("test_overall_metrics"), payload.get("test_subset_metrics") or {}


def normalize_result(method, rate, seed, payload, log_path):
    overall, subsets = metric_sources(method, payload)
    if overall is None:
        raise KeyError("Missing overall metrics for %s rate=%s seed=%s." % (method, rate, seed))
    if MISSING_TEXT_SPLIT not in subsets:
        raise KeyError("Missing subset '%s' for %s rate=%s seed=%s." % (MISSING_TEXT_SPLIT, method, rate, seed))
    return {
        "dataset": DATASET,
        "missing_rate": float(rate),
        "method": method,
        "seed": int(seed),
        "mrr": float(overall["mrr"]),
        "hit1": float(overall["hit1"]),
        "hit3": float(overall["hit3"]),
        "hit10": float(overall["hit10"]),
        "missing_text_mrr": float(subsets[MISSING_TEXT_SPLIT]["mrr"]),
        "mask_checksum_sha256": injection_checksum(payload),
        "log_file": os.path.normpath(log_path),
    }


def load_completed_results(args):
    all_results = []
    for rate in args.rates:
        tag = rate_tag(rate)
        result_dir = os.path.join(args.result_root, tag)
        for seed in args.seeds:
            dhns_log = os.path.join(result_dir, "dhns_seed%d.log" % seed)
            mma_log = os.path.join(result_dir, "mma_hns_seed%d.log" % seed)
            dhns_payload = load_result_json_from_log(dhns_log)
            mma_payload = load_result_json_from_log(mma_log)
            if dhns_payload is None or mma_payload is None:
                raise RuntimeError(
                    "Incomplete paired logs for rate=%s seed=%s. Expected %s and %s."
                    % (rate, seed, dhns_log, mma_log)
                )
            verify_shared_mask(rate, seed, [dhns_payload, mma_payload])
            all_results.append(normalize_result("DHNS", rate, seed, dhns_payload, dhns_log))
            all_results.append(normalize_result("MMA-HNS", rate, seed, mma_payload, mma_log))
    return all_results


def mean_std(values):
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean_value), float(std_value)


def paired_t_test(dhns_values, mma_values, seeds):
    if len(dhns_values) != len(mma_values) or len(dhns_values) != len(seeds):
        raise ValueError("Paired t-test inputs must have the same length.")
    deltas = [mma - dhns for dhns, mma in zip(dhns_values, mma_values)]
    delta_mean = statistics.mean(deltas)
    delta_std = statistics.stdev(deltas)
    degenerate_constant_difference = False
    if all(abs(delta) <= 1e-15 for delta in deltas):
        statistic, p_value = 0.0, 1.0
    elif delta_std <= max(1e-15, abs(delta_mean) * 1e-12):
        # The paired-t statistic tends to +/- infinity when every paired
        # difference is the same non-zero value. Keep the JSON standards-
        # compliant by recording a null statistic and the limiting p-value.
        statistic, p_value = None, 0.0
        degenerate_constant_difference = True
    else:
        test_result = ttest_rel(mma_values, dhns_values, alternative="two-sided")
        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)
    if (statistic is not None and not math.isfinite(statistic)) or not math.isfinite(p_value):
        raise RuntimeError("Paired t-test produced a non-finite result.")
    return {
        "test": "two-sided paired t-test",
        "metric": "overall seed-level MRR",
        "comparison": "MMA-HNS vs DHNS",
        "paired_seed_count": len(seeds),
        "paired_seeds": [int(seed) for seed in seeds],
        "dhns_mrr": dhns_values,
        "mma_hns_mrr": mma_values,
        "mma_hns_minus_dhns": deltas,
        "t_statistic": statistic,
        "p_value": p_value,
        "degenerate_constant_difference": degenerate_constant_difference,
    }


def build_summary(all_results, rates, seeds):
    table_rows = []
    significance_tests = []
    for rate in rates:
        rate_results = [row for row in all_results if abs(row["missing_rate"] - rate) <= 1e-12]
        by_method = {method: [row for row in rate_results if row["method"] == method] for method in METHODS}
        for method in METHODS:
            by_method[method].sort(key=lambda row: row["seed"])
            actual_seeds = [row["seed"] for row in by_method[method]]
            if actual_seeds != sorted(seeds):
                raise RuntimeError("Incomplete seeds for %s at missing rate %s." % (method, rate))

        dhns_mrr = [row["mrr"] for row in by_method["DHNS"]]
        mma_mrr = [row["mrr"] for row in by_method["MMA-HNS"]]
        significance = paired_t_test(dhns_mrr, mma_mrr, sorted(seeds))
        significance["missing_rate"] = float(rate)
        significance_tests.append(significance)

        for method in METHODS:
            method_results = by_method[method]
            summary = {
                "dataset": DATASET,
                "missing_rate": float(rate),
                "method": method,
                "seed_count": len(method_results),
            }
            for metric_key, _label in METRICS:
                summary[metric_key + "_mean"], summary[metric_key + "_std"] = mean_std(
                    [row[metric_key] for row in method_results]
                )
            summary["missing_text_mrr_mean"], summary["missing_text_mrr_std"] = mean_std(
                [row["missing_text_mrr"] for row in method_results]
            )
            summary["paired_mrr_p_value"] = significance["p_value"] if method == "MMA-HNS" else None
            table_rows.append(summary)
    return table_rows, significance_tests


def format_mean_std(mean_value, std_value):
    return "%.6f ± %.6f" % (mean_value, std_value)


def write_runs_csv(path, all_results):
    fieldnames = [
        "dataset",
        "missing_rate",
        "method",
        "seed",
        "mrr",
        "hit1",
        "hit3",
        "hit10",
        "missing_text_mrr",
        "mask_checksum_sha256",
        "log_file",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)


def write_summary_csv(path, table_rows):
    fieldnames = [
        "dataset",
        "missing_rate",
        "method",
        "seed_count",
        "mrr_mean",
        "mrr_std",
        "hit1_mean",
        "hit1_std",
        "hit3_mean",
        "hit3_std",
        "hit10_mean",
        "hit10_std",
        "missing_text_mrr_mean",
        "missing_text_mrr_std",
        "paired_mrr_p_value",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_markdown(path, table_rows):
    lines = [
        "| Missing rate | Method | MRR | H@1 | H@3 | H@10 | Missing-text MRR | p-value |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        p_value = "—" if row["paired_mrr_p_value"] is None else "%.6g" % row["paired_mrr_p_value"]
        lines.append(
            "| %.0f%% | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["missing_rate"] * 100.0,
                row["method"],
                format_mean_std(row["mrr_mean"], row["mrr_std"]),
                format_mean_std(row["hit1_mean"], row["hit1_std"]),
                format_mean_std(row["hit3_mean"], row["hit3_std"]),
                format_mean_std(row["hit10_mean"], row["hit10_std"]),
                format_mean_std(row["missing_text_mrr_mean"], row["missing_text_mrr_std"]),
                p_value,
            )
        )
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


def save_outputs(args, all_results):
    all_results.sort(key=lambda row: (row["missing_rate"], METHODS.index(row["method"]), row["seed"]))
    table_rows, significance_tests = build_summary(all_results, args.rates, args.seeds)
    os.makedirs(args.result_root, exist_ok=True)
    runs_csv = os.path.join(args.result_root, "main_table_runs.csv")
    summary_csv = os.path.join(args.result_root, "main_table_summary.csv")
    markdown_path = os.path.join(args.result_root, "main_table.md")
    result_json_path = os.path.join(args.result_root, "main_table_result.json")

    write_runs_csv(runs_csv, all_results)
    write_summary_csv(summary_csv, table_rows)
    write_markdown(markdown_path, table_rows)
    result_payload = {
        "dataset": DATASET,
        "missing_rates": [float(rate) for rate in args.rates],
        "seeds": [int(seed) for seed in args.seeds],
        "methods": list(METHODS),
        "missing_text_metric_definition": MISSING_TEXT_SPLIT + " MRR",
        "significance_test_scope": "overall seed-level MRR only; no tests for H@1/H@3/H@10",
        "runs": all_results,
        "table_rows": table_rows,
        "significance_tests": significance_tests,
        "output_files": {
            "runs_csv": os.path.abspath(runs_csv),
            "summary_csv": os.path.abspath(summary_csv),
            "markdown_table": os.path.abspath(markdown_path),
            "result_json": os.path.abspath(result_json_path),
        },
    }
    write_json(result_json_path, result_payload)
    print("\nSaved run-level results: %s" % os.path.abspath(runs_csv), flush=True)
    print("Saved summary: %s" % os.path.abspath(summary_csv), flush=True)
    print("Saved paper table: %s" % os.path.abspath(markdown_path), flush=True)
    print("Saved RESULT_JSON: %s" % os.path.abspath(result_json_path), flush=True)
    print("RESULT_JSON: " + json.dumps(result_payload, ensure_ascii=False, sort_keys=True), flush=True)


def main():
    args = parse_args()
    validate_args(args)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    args.result_root = os.path.normpath(args.result_root)
    args.checkpoint_root = os.path.normpath(args.checkpoint_root)
    args.mask_root = os.path.normpath(args.mask_root)

    if args.summary_only:
        all_results = load_completed_results(args)
    else:
        all_results = []
        for rate in args.rates:
            for seed in args.seeds:
                paired_results = run_rate_seed(args, rate, seed)
                all_results.extend([paired_results[method] for method in METHODS])
    save_outputs(args, all_results)


if __name__ == "__main__":
    main()
