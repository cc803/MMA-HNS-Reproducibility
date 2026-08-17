import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from eval_vgse import (
    build_filtered_maps,
    calc_filtered_rank,
    load_checkpoint_compatible,
    make_candidate_batches,
    make_model,
    parse_alpha_grid,
    predict_score,
    print_metrics,
    read_count,
    read_triples,
    apply_text_missing_injection,
    apply_simulated_native_text_missing,
    summarize_missingness,
    summarize_ranks,
)


FALLBACK_LAMBDA = 0.25
FALLBACK_ALPHA = 0.0
LEVELS = ("level1", "level2", "level3")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hierarchical Pareto-Safe Adaptive Calibration for two trained AdvMixRotatE checkpoints."
    )
    parser.add_argument("--dataset", type=str, default="MKG-Y")
    parser.add_argument("--checkpoint-a", type=str, required=True)
    parser.add_argument("--checkpoint-b", type=str, required=True)
    parser.add_argument("--lambda-grid", type=str, default="0.10,0.20,0.25,0.30,0.40")
    parser.add_argument("--alpha-grid", type=str, default="0.0,0.1,0.2,0.3")
    parser.add_argument(
        "--fallback-lambda",
        type=float,
        default=FALLBACK_LAMBDA,
        help="Fallback retrieval mix weight used by HPSAC. Default 0.25 preserves previous results.",
    )
    parser.add_argument("--min-group-queries", type=int, default=30)
    parser.add_argument("--safe-delta", type=float, default=0.0002)
    parser.add_argument("--lock-missing-text", action="store_true")
    parser.add_argument(
        "--separate-calibration-split",
        action="store_true",
        help=(
            "Randomly split validation triples 50/50: reserve the first half for upstream "
            "hyperparameter selection and use only the second half for HPSAC calibration. "
            "Disabled by default to preserve previous results."
        ),
    )
    parser.add_argument(
        "--validation-split-seed",
        type=int,
        default=0,
        help="Random seed for --separate-calibration-split. The default is 0.",
    )
    parser.add_argument(
        "--calibration-mode",
        type=str,
        choices=["hpsac", "global", "group_no_safety", "fixed", "alpha_only", "temperature"],
        default="hpsac",
        help="Calibration policy used at evaluation time. The default hpsac preserves previous behavior.",
    )
    parser.add_argument(
        "--temperature-grid",
        type=str,
        default="0.50,0.75,1.00,1.25,1.50,2.00",
        help="Positive temperature candidates for --calibration-mode temperature.",
    )
    parser.add_argument("--subset-eval", action="store_true")
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=512)
    parser.add_argument(
        "--retrieval-source",
        type=str,
        choices=["entity_embedding_knn", "random_text_pool"],
        default="entity_embedding_knn",
    )
    parser.add_argument(
        "--inject-text-missing-rate",
        type=float,
        default=0.0,
        help="Apply the same artificial text masking protocol as train_dhns_rotate.py before evaluation.",
    )
    parser.add_argument(
        "--text-missing-mask-strategy",
        type=str,
        choices=["random", "low_degree", "high_degree"],
        default="random",
        help="Entity-level artificial text-missing mask strategy. The default random setting preserves previous behavior.",
    )
    parser.add_argument(
        "--text-missing-mask-path",
        type=str,
        default=None,
        help="Optional fixed entity-level text-missing mask file (.pt/.pth/.json) to reuse across methods.",
    )
    parser.add_argument(
        "--mask-file",
        dest="text_missing_mask_path",
        type=str,
        default=None,
        help="Alias for --text-missing-mask-path.",
    )
    parser.add_argument(
        "--text-missing-mask-seed",
        type=int,
        default=0,
        help="Seed used when generating an evaluation-time text-missing mask without --text-missing-mask-path.",
    )
    parser.add_argument(
        "--save-text-missing-mask-path",
        type=str,
        default=None,
        help="Optional path for saving the generated or loaded entity-level text-missing mask for reproducibility.",
    )
    parser.add_argument(
        "--simulate-native-text-missing-rate",
        type=float,
        default=0.0,
        help="Zero a fraction of originally available text embeddings before missingness is summarized, matching train_dhns_rotate.py controlled native-like diagnostics.",
    )
    parser.add_argument("--no-gpu", action="store_true")
    return parser.parse_args()


def validation_index_checksum(indices):
    encoded = ",".join(str(int(index)) for index in indices).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_positive_grid(raw_values, arg_name):
    values = []
    for raw_value in raw_values.split(","):
        stripped = raw_value.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value <= 0.0:
            raise ValueError("%s values must be > 0." % arg_name)
        values.append(value)
    if not values:
        raise ValueError("%s must contain at least one value." % arg_name)
    return values


def split_validation_for_calibration(valid_triples, enabled=False, seed=0):
    original_count = len(valid_triples)
    if original_count == 0:
        raise ValueError("Validation set is empty.")

    if enabled:
        permutation = np.random.default_rng(seed).permutation(original_count).tolist()
        split_index = original_count // 2
        hyperparameter_indices = permutation[:split_index]
        calibration_indices = permutation[split_index:]
        mode = "random_50_50_separate_calibration"
    else:
        hyperparameter_indices = []
        calibration_indices = list(range(original_count))
        mode = "full_validation_legacy"

    hyperparameter_triples = [valid_triples[index] for index in hyperparameter_indices]
    calibration_triples = [valid_triples[index] for index in calibration_indices]
    split_info = {
        "mode": mode,
        "separate_calibration_split": bool(enabled),
        "validation_split_seed": int(seed),
        "random_generator": "numpy.random.default_rng(PCG64)",
        "original_validation_triple_count": int(original_count),
        "hyperparameter_selection_triple_count": int(len(hyperparameter_triples)),
        "hpsac_calibration_triple_count": int(len(calibration_triples)),
        "hyperparameter_selection_index_checksum_sha256": validation_index_checksum(hyperparameter_indices),
        "hpsac_calibration_index_checksum_sha256": validation_index_checksum(calibration_indices),
        "hyperparameter_selection_indices_preview": [int(index) for index in hyperparameter_indices[:20]],
        "hpsac_calibration_indices_preview": [int(index) for index in calibration_indices[:20]],
        "partitions_disjoint": len(set(hyperparameter_indices) & set(calibration_indices)) == 0,
        "partitions_cover_validation": (
            len(hyperparameter_indices) + len(calibration_indices) == original_count
            and set(hyperparameter_indices) | set(calibration_indices) == set(range(original_count))
        ),
        "hyperparameter_partition_usage": (
            "reserved_for_upstream_epoch_learning_rate_and_base_model_selection"
            if enabled else "not_separated_in_legacy_mode"
        ),
        "hpsac_partition_usage": "groupwise_lambda_alpha_selection_and_safety_checks",
    }
    return hyperparameter_triples, calibration_triples, split_info


def make_config(lambda_value, alpha):
    return {
        "lambda": float(lambda_value),
        "alpha": float(alpha),
        "key": "%.6f|%.6f" % (float(lambda_value), float(alpha)),
    }


def make_candidate_configs(lambda_grid, alpha_grid):
    return [make_config(lambda_value, alpha) for lambda_value in lambda_grid for alpha in alpha_grid]


def fallback_config():
    return make_config(FALLBACK_LAMBDA, FALLBACK_ALPHA)


def config_is_fallback(config):
    return (
        abs(float(config["lambda"]) - FALLBACK_LAMBDA) < 1e-12 and
        abs(float(config["alpha"]) - FALLBACK_ALPHA) < 1e-12
    )


def all_eval_configs(candidate_configs):
    configs = list(candidate_configs)
    fallback = fallback_config()
    if not any(config["key"] == fallback["key"] for config in configs):
        configs.append(fallback)
    return configs


def text_state_for_triple(h, t, has_text_np):
    return "both_have_text" if bool(has_text_np[h]) and bool(has_text_np[t]) else "missing_text"


def group_keys(relation_id, prediction_side, text_state):
    return {
        "level1": (int(relation_id), prediction_side, text_state),
        "level2": (prediction_side, text_state),
        "level3": (text_state,),
    }


def group_to_string(level, key):
    if level == "level1":
        relation_id, prediction_side, text_state = key
        return "relation=%s|side=%s|text=%s" % (relation_id, prediction_side, text_state)
    if level == "level2":
        prediction_side, text_state = key
        return "side=%s|text=%s" % (prediction_side, text_state)
    text_state = key[0]
    return "text=%s" % text_state


def group_text_state(key):
    return key[-1]


def empty_eval_state(subset_eval):
    groups = {}
    if subset_eval:
        groups = {
            "head_missing_text": {"_ranks": [], "triple_count": 0},
            "tail_missing_text": {"_ranks": [], "triple_count": 0},
            "head_or_tail_missing_text": {"_ranks": [], "triple_count": 0},
            "head_and_tail_have_text": {"_ranks": [], "triple_count": 0},
        }
    return {"overall_ranks": [], "groups": groups}


def update_subset_state(state, ranks, h, t, has_text_np):
    if not state["groups"]:
        return
    head_has_text = bool(has_text_np[h])
    tail_has_text = bool(has_text_np[t])
    if not head_has_text:
        state["groups"]["head_missing_text"]["_ranks"].extend(ranks)
        state["groups"]["head_missing_text"]["triple_count"] += 1
    if not tail_has_text:
        state["groups"]["tail_missing_text"]["_ranks"].extend(ranks)
        state["groups"]["tail_missing_text"]["triple_count"] += 1
    if (not head_has_text) or (not tail_has_text):
        group_name = "head_or_tail_missing_text"
    else:
        group_name = "head_and_tail_have_text"
    state["groups"][group_name]["_ranks"].extend(ranks)
    state["groups"][group_name]["triple_count"] += 1


def finalize_eval_state(state):
    overall = summarize_ranks(state["overall_ranks"])
    overall["count_type"] = "query"
    overall["triple_count"] = int(len(state["overall_ranks"]) // 2)
    subset_metrics = {}
    for group_name, group_state in state["groups"].items():
        summary = summarize_ranks(group_state["_ranks"])
        summary["count_type"] = "query"
        summary["triple_count"] = int(group_state["triple_count"])
        subset_metrics[group_name] = summary
    if "head_and_tail_have_text" in subset_metrics:
        subset_metrics["head_or_tail_both_have_text"] = dict(subset_metrics["head_and_tail_have_text"])
    return overall, subset_metrics if subset_metrics else None


def get_b_score(model_b, data, lambda_value, use_gpu, cache):
    key = float(lambda_value)
    if key not in cache:
        model_b.retrieval_mix_weight = key
        cache[key] = predict_score(model_b, data, use_gpu)
    return cache[key]


def add_group_rank(group_stats, relation_id, prediction_side, text_state, config_key, rank):
    for level, key in group_keys(relation_id, prediction_side, text_state).items():
        group_stats[level][key][config_key].append(rank)


def collect_validation_group_stats(
    model_a,
    model_b,
    triples,
    hr_to_tails,
    tr_to_heads,
    has_text,
    configs,
    use_gpu,
    lock_missing_text=False,
):
    group_stats = {level: defaultdict(lambda: defaultdict(list)) for level in LEVELS}
    has_text_np = has_text.cpu().numpy().astype(bool)
    for h, t, r in tqdm(triples, desc="HPSAC valid search", leave=False):
        head_data, tail_data = make_candidate_batches(h, t, r, model_a.ent_tot)
        text_state = text_state_for_triple(h, t, has_text_np)
        if lock_missing_text and text_state == "missing_text":
            fallback = fallback_config()
            configs_for_query = [fallback]
        else:
            configs_for_query = configs

        head_a = predict_score(model_a, head_data, use_gpu)
        tail_a = predict_score(model_a, tail_data, use_gpu)
        head_b_cache = {}
        tail_b_cache = {}
        for config in configs_for_query:
            lambda_value = config["lambda"]
            alpha = config["alpha"]
            head_b = get_b_score(model_b, head_data, lambda_value, use_gpu, head_b_cache)
            tail_b = get_b_score(model_b, tail_data, lambda_value, use_gpu, tail_b_cache)
            head_score = (1.0 - alpha) * head_b + alpha * head_a
            tail_score = (1.0 - alpha) * tail_b + alpha * tail_a
            head_rank = calc_filtered_rank(head_score, h, tr_to_heads[(t, r)] - {h})
            tail_rank = calc_filtered_rank(tail_score, t, hr_to_tails[(h, r)] - {t})
            add_group_rank(group_stats, r, "head_prediction", text_state, config["key"], head_rank)
            add_group_rank(group_stats, r, "tail_prediction", text_state, config["key"], tail_rank)
    return group_stats


def metrics_for_ranks(ranks):
    summary = summarize_ranks(ranks)
    return {
        "query_count": summary["count"],
        "mrr": summary["mrr"],
        "mr": summary["mr"],
        "hit10": summary["hit10"],
        "hit3": summary["hit3"],
        "hit1": summary["hit1"],
    }


def group_selection_key(candidate):
    metrics = candidate["metrics"]
    return (
        metrics["mrr"],
        metrics["hit10"],
        metrics["hit3"],
        metrics["hit1"],
        -metrics["mr"],
        -candidate["alpha"],
        -candidate["lambda"],
    )


def select_group_configs(
    group_stats,
    candidate_configs,
    min_group_queries,
    safe_delta,
    lock_missing_text=False,
    enforce_safety=True,
):
    fallback = fallback_config()
    fallback_key = fallback["key"]
    selected = {level: {} for level in LEVELS}
    records = {level: {} for level in LEVELS}
    for level in LEVELS:
        for key, config_ranks in group_stats[level].items():
            fallback_metrics = metrics_for_ranks(config_ranks[fallback_key])
            query_count = int(fallback_metrics["query_count"])
            base_record = {
                "level": level,
                "group": group_to_string(level, key),
                "query_count": query_count,
                "text_state": group_text_state(key),
                "fallback_lambda": FALLBACK_LAMBDA,
                "fallback_alpha": FALLBACK_ALPHA,
                "fallback_valid_metrics": fallback_metrics,
            }
            if lock_missing_text and group_text_state(key) == "missing_text":
                entry = {
                    **base_record,
                    "selected_lambda": FALLBACK_LAMBDA,
                    "selected_alpha": FALLBACK_ALPHA,
                    "selected_valid_metrics": fallback_metrics,
                    "valid_mrr_delta_vs_bv1": 0.0,
                    "accepted": False,
                    "reason": "missing_text_locked_to_bv1",
                    "lock_missing_text": True,
                }
                selected[level][key] = entry
                records[level][key] = entry
                continue
            if query_count < min_group_queries:
                records[level][key] = {
                    **base_record,
                    "selected_lambda": None,
                    "selected_alpha": None,
                    "accepted": False,
                    "reason": "insufficient_validation_queries",
                }
                continue

            candidates = []
            for config in candidate_configs:
                ranks = config_ranks.get(config["key"], [])
                if len(ranks) == 0:
                    continue
                metrics = metrics_for_ranks(ranks)
                candidates.append(
                    {
                        "lambda": config["lambda"],
                        "alpha": config["alpha"],
                        "key": config["key"],
                        "metrics": metrics,
                    }
                )
            best = max(candidates, key=group_selection_key) if candidates else None
            accepted = False
            reason = "no_candidate"
            if best is not None:
                if enforce_safety:
                    mrr_gain = best["metrics"]["mrr"] - fallback_metrics["mrr"]
                    accepted = mrr_gain >= safe_delta
                    reason = "accepted" if accepted else "pareto_safe_rejected"
                    if group_text_state(key) == "missing_text" and best["metrics"]["mrr"] < fallback_metrics["mrr"]:
                        accepted = False
                        reason = "missing_text_mrr_below_bv1"
                else:
                    accepted = not config_is_fallback(best)
                    reason = "accepted_no_safety" if accepted else "validation_best_is_fallback"

            if accepted:
                entry = {
                    **base_record,
                    "selected_lambda": best["lambda"],
                    "selected_alpha": best["alpha"],
                    "selected_valid_metrics": best["metrics"],
                    "valid_mrr_delta_vs_bv1": best["metrics"]["mrr"] - fallback_metrics["mrr"],
                    "accepted": True,
                    "reason": reason,
                }
            else:
                entry = {
                    **base_record,
                    "selected_lambda": FALLBACK_LAMBDA,
                    "selected_alpha": FALLBACK_ALPHA,
                    "selected_valid_metrics": fallback_metrics,
                    "valid_mrr_delta_vs_bv1": 0.0,
                    "accepted": False,
                    "reason": reason,
                }
            selected[level][key] = entry
            records[level][key] = entry
    return selected, records


def select_global_config(group_stats, candidate_configs):
    fallback = fallback_config()
    fallback_key = fallback["key"]
    candidates = []
    for config in candidate_configs:
        ranks = []
        for config_ranks in group_stats["level3"].values():
            ranks.extend(config_ranks.get(config["key"], []))
        if not ranks:
            continue
        candidates.append(
            {
                "lambda": config["lambda"],
                "alpha": config["alpha"],
                "key": config["key"],
                "metrics": metrics_for_ranks(ranks),
            }
        )
    if not candidates:
        raise ValueError("No global interpolation candidates were evaluated.")

    fallback_ranks = []
    for config_ranks in group_stats["level3"].values():
        fallback_ranks.extend(config_ranks.get(fallback_key, []))
    fallback_metrics = metrics_for_ranks(fallback_ranks)
    best = max(candidates, key=group_selection_key)
    accepted = not config_is_fallback(best)
    reason = "global_validation_best" if accepted else "global_validation_best_is_fallback"
    entry = {
        "level": "global",
        "group": "global",
        "query_count": int(best["metrics"]["query_count"]),
        "text_state": "all",
        "fallback_lambda": FALLBACK_LAMBDA,
        "fallback_alpha": FALLBACK_ALPHA,
        "fallback_valid_metrics": fallback_metrics,
        "selected_lambda": best["lambda"],
        "selected_alpha": best["alpha"],
        "selected_valid_metrics": best["metrics"],
        "valid_mrr_delta_vs_bv1": best["metrics"]["mrr"] - fallback_metrics["mrr"],
        "accepted": accepted,
        "reason": reason,
    }
    selected = {level: {} for level in LEVELS}
    for key in group_stats["level3"].keys():
        selected["level3"][key] = dict(entry)
    records = {level: {} for level in LEVELS}
    records["level3"][("global",)] = entry
    return selected, records, entry


def select_fixed_config(group_stats):
    fallback = fallback_config()
    fallback_key = fallback["key"]
    selected = {level: {} for level in LEVELS}
    records = {level: {} for level in LEVELS}
    for level in LEVELS:
        for key, config_ranks in group_stats[level].items():
            fallback_metrics = metrics_for_ranks(config_ranks[fallback_key])
            entry = {
                "level": level,
                "group": group_to_string(level, key),
                "query_count": int(fallback_metrics["query_count"]),
                "text_state": group_text_state(key),
                "fallback_lambda": FALLBACK_LAMBDA,
                "fallback_alpha": FALLBACK_ALPHA,
                "fallback_valid_metrics": fallback_metrics,
                "selected_lambda": FALLBACK_LAMBDA,
                "selected_alpha": FALLBACK_ALPHA,
                "selected_valid_metrics": fallback_metrics,
                "valid_mrr_delta_vs_bv1": 0.0,
                "accepted": False,
                "reason": "fixed_default_parameters",
                "group_search_enabled": False,
            }
            selected[level][key] = entry
            records[level][key] = entry
    return selected, records


def temperature_selection_key(candidate):
    metrics = candidate["metrics"]
    temperature = float(candidate["temperature"])
    return (
        metrics["mrr"],
        metrics["hit10"],
        metrics["hit3"],
        metrics["hit1"],
        -metrics["mr"],
        -abs(temperature - 1.0),
        -temperature,
    )


def select_temperature_config(group_stats, temperature_grid):
    fallback = fallback_config()
    fallback_key = fallback["key"]
    fallback_ranks = []
    for config_ranks in group_stats["level3"].values():
        fallback_ranks.extend(config_ranks.get(fallback_key, []))
    if not fallback_ranks:
        raise ValueError("No validation ranks are available for temperature calibration.")

    candidates = []
    for temperature in temperature_grid:
        metrics = metrics_for_ranks(fallback_ranks)
        candidates.append(
            {
                "temperature": float(temperature),
                "metrics": metrics,
                "rank_invariant_to_positive_temperature": True,
            }
        )
    best = max(candidates, key=temperature_selection_key)
    fallback_metrics = metrics_for_ranks(fallback_ranks)
    entry = {
        "level": "global",
        "group": "global_temperature",
        "query_count": int(best["metrics"]["query_count"]),
        "text_state": "all",
        "fallback_lambda": FALLBACK_LAMBDA,
        "fallback_alpha": FALLBACK_ALPHA,
        "fallback_temperature": 1.0,
        "fallback_valid_metrics": fallback_metrics,
        "selected_lambda": FALLBACK_LAMBDA,
        "selected_alpha": FALLBACK_ALPHA,
        "selected_temperature": best["temperature"],
        "selected_valid_metrics": best["metrics"],
        "valid_mrr_delta_vs_bv1": best["metrics"]["mrr"] - fallback_metrics["mrr"],
        "accepted": abs(best["temperature"] - 1.0) > 1e-12,
        "reason": "temperature_validation_search_rank_invariant",
        "temperature_candidate_metrics": candidates,
        "rank_invariant_to_positive_temperature": True,
    }
    selected = {level: {} for level in LEVELS}
    records = {level: {} for level in LEVELS}
    for key in group_stats["level3"].keys():
        selected["level3"][key] = dict(entry)
        selected["level3"][key]["group"] = group_to_string("level3", key)
        records["level3"][key] = dict(selected["level3"][key])
    return selected, records, entry


def resolve_entry(selected, relation_id, prediction_side, text_state, lock_missing_text=False):
    if lock_missing_text and text_state == "missing_text":
        return {
            "selected_lambda": FALLBACK_LAMBDA,
            "selected_alpha": FALLBACK_ALPHA,
            "accepted": False,
            "reason": "missing_text_locked_to_bv1",
            "resolved_level": "level4",
            "resolved_group": "missing_text_lock_fallback",
        }
    keys = group_keys(relation_id, prediction_side, text_state)
    for level in ("level1", "level2", "level3"):
        key = keys[level]
        if key in selected[level]:
            entry = selected[level][key]
            return {
                **entry,
                "resolved_level": level,
                "resolved_group": group_to_string(level, key),
            }
    return {
        "selected_lambda": FALLBACK_LAMBDA,
        "selected_alpha": FALLBACK_ALPHA,
        "accepted": False,
        "reason": "global_fallback",
        "resolved_level": "level4",
        "resolved_group": "global_fallback",
    }


def score_with_entry(model_a, model_b, data, entry, score_a, use_gpu, cache):
    lambda_value = float(entry["selected_lambda"])
    alpha = float(entry["selected_alpha"])
    temperature = float(entry.get("selected_temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("selected_temperature must be > 0.")
    score_b = get_b_score(model_b, data, lambda_value, use_gpu, cache)
    return ((1.0 - alpha) * score_b + alpha * score_a) / temperature


def policy_count_key(entry):
    key = "%s|lambda=%.2f|alpha=%.1f" % (
        entry["resolved_level"],
        entry["selected_lambda"],
        entry["selected_alpha"],
    )
    if "selected_temperature" in entry:
        key += "|T=%.4g" % float(entry["selected_temperature"])
    return key


def evaluate_policy_and_fallback(
    model_a,
    model_b,
    triples,
    hr_to_tails,
    tr_to_heads,
    has_text,
    selected,
    use_gpu,
    subset_eval,
    split_name,
    lock_missing_text=False,
):
    selected_state = empty_eval_state(subset_eval)
    fallback_state = empty_eval_state(subset_eval)
    policy_counts = Counter()
    has_text_np = has_text.cpu().numpy().astype(bool)
    fallback_entry = {
        "selected_lambda": FALLBACK_LAMBDA,
        "selected_alpha": FALLBACK_ALPHA,
        "resolved_level": "level4",
        "resolved_group": "global_fallback",
    }
    for h, t, r in tqdm(triples, desc=f"HPSAC {split_name}", leave=False):
        head_data, tail_data = make_candidate_batches(h, t, r, model_a.ent_tot)
        text_state = text_state_for_triple(h, t, has_text_np)

        head_a = predict_score(model_a, head_data, use_gpu)
        tail_a = predict_score(model_a, tail_data, use_gpu)
        head_b_cache = {}
        tail_b_cache = {}
        head_entry = resolve_entry(selected, r, "head_prediction", text_state, lock_missing_text=lock_missing_text)
        tail_entry = resolve_entry(selected, r, "tail_prediction", text_state, lock_missing_text=lock_missing_text)
        policy_counts[policy_count_key(head_entry)] += 1
        policy_counts[policy_count_key(tail_entry)] += 1

        selected_head_score = score_with_entry(model_a, model_b, head_data, head_entry, head_a, use_gpu, head_b_cache)
        selected_tail_score = score_with_entry(model_a, model_b, tail_data, tail_entry, tail_a, use_gpu, tail_b_cache)
        fallback_head_score = score_with_entry(model_a, model_b, head_data, fallback_entry, head_a, use_gpu, head_b_cache)
        fallback_tail_score = score_with_entry(model_a, model_b, tail_data, fallback_entry, tail_a, use_gpu, tail_b_cache)

        head_filtered = tr_to_heads[(t, r)] - {h}
        tail_filtered = hr_to_tails[(h, r)] - {t}
        selected_ranks = [
            calc_filtered_rank(selected_head_score, h, head_filtered),
            calc_filtered_rank(selected_tail_score, t, tail_filtered),
        ]
        fallback_ranks = [
            calc_filtered_rank(fallback_head_score, h, head_filtered),
            calc_filtered_rank(fallback_tail_score, t, tail_filtered),
        ]
        selected_state["overall_ranks"].extend(selected_ranks)
        fallback_state["overall_ranks"].extend(fallback_ranks)
        update_subset_state(selected_state, selected_ranks, h, t, has_text_np)
        update_subset_state(fallback_state, fallback_ranks, h, t, has_text_np)

    selected_overall, selected_subset = finalize_eval_state(selected_state)
    fallback_overall, fallback_subset = finalize_eval_state(fallback_state)
    return {
        "selected_overall": selected_overall,
        "selected_subset": selected_subset,
        "bv1_overall": fallback_overall,
        "bv1_subset": fallback_subset,
        "policy_counts": dict(policy_counts),
    }


def metric_delta(selected_metrics, fallback_metrics):
    return {
        "mrr": selected_metrics["mrr"] - fallback_metrics["mrr"],
        "mr": selected_metrics["mr"] - fallback_metrics["mr"],
        "hit10": selected_metrics["hit10"] - fallback_metrics["hit10"],
        "hit3": selected_metrics["hit3"] - fallback_metrics["hit3"],
        "hit1": selected_metrics["hit1"] - fallback_metrics["hit1"],
    }


def subset_delta(selected_subset, fallback_subset, group_name):
    if selected_subset is None or fallback_subset is None:
        return None
    if group_name not in selected_subset or group_name not in fallback_subset:
        return None
    return metric_delta(selected_subset[group_name], fallback_subset[group_name])


def serialize_group_records(records):
    output = {}
    for level in LEVELS:
        output[level] = {}
        for key, record in records[level].items():
            output[level][group_to_string(level, key)] = record
    return output


def config_uses_default(config):
    return (
        abs(float(config["selected_lambda"]) - FALLBACK_LAMBDA) <= 1e-12 and
        abs(float(config["selected_alpha"]) - FALLBACK_ALPHA) <= 1e-12 and
        abs(float(config.get("selected_temperature", 1.0)) - 1.0) <= 1e-12
    )


def build_hpsac_stats(records, effective_level1_configs, min_group_queries, safe_delta):
    level_stats = {}
    total_group_count = 0
    total_accepted_count = 0
    for level in LEVELS:
        level_records = list(records[level].values())
        reason_counts = Counter(record.get("reason", "unknown") for record in level_records)
        accepted_count = sum(1 for record in level_records if bool(record.get("accepted")))
        fallback_count = len(level_records) - accepted_count
        total_group_count += len(level_records)
        total_accepted_count += accepted_count
        level_stats[level] = {
            "total_group_count": int(len(level_records)),
            "specific_parameter_group_count": int(accepted_count),
            "accepted_specific_parameter_group_count": int(accepted_count),
            "fallback_group_count": int(fallback_count),
            "fallback_default_group_count": int(fallback_count),
            "accepted_ratio": float(accepted_count / len(level_records)) if level_records else 0.0,
            "fallback_ratio": float(fallback_count / len(level_records)) if level_records else 0.0,
            "insufficient_validation_queries_group_count": int(reason_counts.get("insufficient_validation_queries", 0)),
            "pareto_safe_rejected_group_count": int(reason_counts.get("pareto_safe_rejected", 0)),
            "missing_text_mrr_below_bv1_group_count": int(reason_counts.get("missing_text_mrr_below_bv1", 0)),
            "missing_text_locked_group_count": int(reason_counts.get("missing_text_locked_to_bv1", 0)),
            "no_candidate_group_count": int(reason_counts.get("no_candidate", 0)),
            "reason_counts": {reason: int(count) for reason, count in sorted(reason_counts.items())},
        }

    effective_records = list(effective_level1_configs.values())
    effective_specific_count = sum(1 for entry in effective_records if not config_uses_default(entry))
    effective_default_count = len(effective_records) - effective_specific_count
    resolved_level_counts = Counter(entry.get("resolved_level", "unknown") for entry in effective_records)
    effective_reason_counts = Counter(entry.get("reason", "unknown") for entry in effective_records)
    return {
        "fallback_lambda": FALLBACK_LAMBDA,
        "fallback_alpha": FALLBACK_ALPHA,
        "min_group_queries_name": "N_min",
        "min_group_queries": int(min_group_queries),
        "safe_delta_name": "delta",
        "safe_delta": float(safe_delta),
        "levels": level_stats,
        "all_levels_total_group_count": int(total_group_count),
        "all_levels_specific_parameter_group_count": int(total_accepted_count),
        "all_levels_accepted_specific_parameter_group_count": int(total_accepted_count),
        "all_levels_fallback_group_count": int(total_group_count - total_accepted_count),
        "all_levels_fallback_default_group_count": int(total_group_count - total_accepted_count),
        "all_levels_accepted_ratio": (
            float(total_accepted_count / total_group_count) if total_group_count > 0 else 0.0
        ),
        "all_levels_fallback_ratio": (
            float((total_group_count - total_accepted_count) / total_group_count)
            if total_group_count > 0 else 0.0
        ),
        "effective_level1_policy": {
            "total_group_count": int(len(effective_records)),
            "specific_parameter_group_count": int(effective_specific_count),
            "default_fallback_group_count": int(effective_default_count),
            "specific_parameter_ratio": (
                float(effective_specific_count / len(effective_records)) if effective_records else 0.0
            ),
            "default_fallback_ratio": (
                float(effective_default_count / len(effective_records)) if effective_records else 0.0
            ),
            "resolved_level_counts": {
                level: int(count)
                for level, count in sorted(resolved_level_counts.items())
            },
            "reason_counts": {
                reason: int(count)
                for reason, count in sorted(effective_reason_counts.items())
            },
        },
    }


def build_effective_level1_configs(group_stats, selected, lock_missing_text=False):
    effective = {}
    for key in sorted(group_stats["level1"].keys()):
        relation_id, prediction_side, text_state = key
        entry = resolve_entry(
            selected,
            relation_id,
            prediction_side,
            text_state,
            lock_missing_text=lock_missing_text,
        )
        effective[group_to_string("level1", key)] = {
            "selected_lambda": entry["selected_lambda"],
            "selected_alpha": entry["selected_alpha"],
            "selected_temperature": float(entry.get("selected_temperature", 1.0)),
            "resolved_level": entry["resolved_level"],
            "resolved_group": entry["resolved_group"],
            "reason": entry["reason"],
            "accepted": bool(entry["accepted"]),
            "text_state": text_state,
        }
    return effective


def count_non_fallback_effective(effective_configs):
    return sum(
        1
        for entry in effective_configs.values()
        if (
            abs(float(entry["selected_lambda"]) - FALLBACK_LAMBDA) > 1e-12 or
            abs(float(entry["selected_alpha"]) - FALLBACK_ALPHA) > 1e-12 or
            abs(float(entry.get("selected_temperature", 1.0)) - 1.0) > 1e-12
        )
    )


def count_non_fallback_both_have_text(effective_configs):
    return sum(
        1
        for entry in effective_configs.values()
        if entry.get("text_state") == "both_have_text" and (
            abs(float(entry["selected_lambda"]) - FALLBACK_LAMBDA) > 1e-12 or
            abs(float(entry["selected_alpha"]) - FALLBACK_ALPHA) > 1e-12 or
            abs(float(entry.get("selected_temperature", 1.0)) - 1.0) > 1e-12
        )
    )


def missing_text_matches_bv1(subset_metrics, bv1_subset_metrics, eps=1e-12):
    if subset_metrics is None or bv1_subset_metrics is None:
        return None
    group_name = "head_or_tail_missing_text"
    if group_name not in subset_metrics or group_name not in bv1_subset_metrics:
        return None
    selected = subset_metrics[group_name]
    fallback = bv1_subset_metrics[group_name]
    return all(
        abs(float(selected[metric_name]) - float(fallback[metric_name])) <= eps
        for metric_name in ["mrr", "mr", "hit10", "hit3", "hit1"]
    )


def main():
    run_start_time = time.perf_counter()
    args = parse_args()
    global FALLBACK_LAMBDA
    if args.fallback_lambda < 0.0 or args.fallback_lambda > 1.0:
        raise ValueError("--fallback-lambda must be in [0, 1].")
    FALLBACK_LAMBDA = float(args.fallback_lambda)
    if args.inject_text_missing_rate > 0.0 and args.simulate_native_text_missing_rate > 0.0:
        raise ValueError("--inject-text-missing-rate and --simulate-native-text-missing-rate cannot be combined.")
    if args.min_group_queries <= 0:
        raise ValueError("--min-group-queries must be > 0.")
    if args.safe_delta < 0.0:
        raise ValueError("--safe-delta must be >= 0.")
    if args.text_missing_mask_path is not None and not os.path.exists(args.text_missing_mask_path):
        raise FileNotFoundError(f"Text-missing mask file not found: {args.text_missing_mask_path}")

    requested_lambda_grid = parse_alpha_grid(args.lambda_grid)
    alpha_grid = parse_alpha_grid(args.alpha_grid)
    temperature_grid = parse_positive_grid(args.temperature_grid, "--temperature-grid")
    if args.calibration_mode in ("fixed", "temperature"):
        lambda_grid = [FALLBACK_LAMBDA]
        effective_alpha_grid = [FALLBACK_ALPHA]
    elif args.calibration_mode == "alpha_only":
        lambda_grid = [FALLBACK_LAMBDA]
        effective_alpha_grid = alpha_grid
    else:
        lambda_grid = requested_lambda_grid
        effective_alpha_grid = alpha_grid
    candidate_configs = make_candidate_configs(lambda_grid, effective_alpha_grid)
    eval_configs = all_eval_configs(candidate_configs)

    benchmark_path = f"./benchmarks/{args.dataset}/"
    visual_path = f"./embeddings/{args.dataset}-visual.pth"
    textual_path = f"./embeddings/{args.dataset}-textual.pth"
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    img_emb = torch.load(visual_path, map_location="cpu")
    text_emb = torch.load(textual_path, map_location="cpu")
    text_emb, simulated_native_text_info, simulated_native_text_mask = apply_simulated_native_text_missing(
        text_emb,
        args.simulate_native_text_missing_rate,
        seed=0,
    )
    text_emb, injection_info, injected_text_mask = apply_text_missing_injection(
        text_emb,
        args.inject_text_missing_rate,
        seed=args.text_missing_mask_seed,
        mask_strategy=args.text_missing_mask_strategy,
        benchmark_path=benchmark_path,
        mask_path=args.text_missing_mask_path,
        save_mask_path=args.save_text_missing_mask_path,
    )
    has_text, has_image = summarize_missingness(img_emb, text_emb)
    ent_tot = int(img_emb.shape[0])
    rel_tot = read_count(os.path.join(benchmark_path, "relation2id.txt"))
    if ent_tot != read_count(os.path.join(benchmark_path, "entity2id.txt")):
        raise ValueError("Embedding entity count does not match entity2id.txt.")

    print(
        "HPSAC config | dataset=%s | lambda_grid=%s | alpha_grid=%s | temperature_grid=%s | min_group_queries=%d | safe_delta=%.6f | calibration_mode=%s"
        % (
            args.dataset,
            ",".join(str(value) for value in lambda_grid),
            ",".join(str(value) for value in effective_alpha_grid),
            ",".join(str(value) for value in temperature_grid),
            args.min_group_queries,
            args.safe_delta,
            args.calibration_mode,
        )
    )
    print(
        "HPSAC validation partition | separate_calibration_split=%s | split_seed=%d"
        % ("True" if args.separate_calibration_split else "False", args.validation_split_seed)
    )
    if simulated_native_text_info is not None:
        print(
            "HPSAC simulated native text missingness | requested=%.4f | applied=%.4f | count=%d"
            % (
                simulated_native_text_info["simulate_rate_requested"],
                simulated_native_text_info["simulate_rate_applied"],
                simulated_native_text_info["simulated_native_missing_count"],
            )
        )
    if injection_info is not None:
        print(
            "HPSAC injected text missingness | requested=%.4f | applied=%.4f | count=%d"
            % (
                injection_info["inject_rate_requested"],
                injection_info["inject_rate_applied"],
                injection_info["additional_masked_count"],
            )
        )
        print(
            "HPSAC text mask | strategy=%s | source=%s | checksum=%s"
            % (
                injection_info["mask_strategy"],
                injection_info["mask_source"],
                injection_info["mask_checksum_sha256"],
            )
        )
    print("HPSAC MT guard | lock_missing_text=%s" % ("True" if args.lock_missing_text else "False"))
    print("HPSAC safety | training=False | new_network=False | internal_scoring_unchanged=True")

    # make_model expects retrieval_mix_weight; HPSAC mutates B only at eval time.
    args.retrieval_mix_weight = FALLBACK_LAMBDA
    model_a = make_model(
        args,
        img_emb,
        text_emb,
        has_text,
        has_image,
        ent_tot,
        rel_tot,
        use_retrieval_missing_text=False,
    )
    model_b = make_model(
        args,
        img_emb,
        text_emb,
        has_text,
        has_image,
        ent_tot,
        rel_tot,
        use_retrieval_missing_text=True,
    )
    load_checkpoint_compatible(model_a, args.checkpoint_a)
    load_checkpoint_compatible(model_b, args.checkpoint_b)
    if use_gpu:
        model_a.cuda()
        model_b.cuda()
    model_a.eval()
    model_b.eval()

    valid_triples = read_triples(os.path.join(benchmark_path, "valid2id.txt"))
    _hyperparameter_valid_triples, calibration_valid_triples, validation_split_info = split_validation_for_calibration(
        valid_triples,
        enabled=args.separate_calibration_split,
        seed=args.validation_split_seed,
    )
    test_triples = read_triples(os.path.join(benchmark_path, "test2id.txt"))
    hr_to_tails, tr_to_heads = build_filtered_maps(benchmark_path)
    print(
        "HPSAC validation split | total=%d | hyperparameter_selection=%d | hpsac_calibration=%d | calibration_checksum=%s"
        % (
            validation_split_info["original_validation_triple_count"],
            validation_split_info["hyperparameter_selection_triple_count"],
            validation_split_info["hpsac_calibration_triple_count"],
            validation_split_info["hpsac_calibration_index_checksum_sha256"],
        )
    )

    group_stats = collect_validation_group_stats(
        model_a,
        model_b,
        calibration_valid_triples,
        hr_to_tails,
        tr_to_heads,
        has_text,
        eval_configs,
        use_gpu,
        lock_missing_text=args.lock_missing_text,
    )
    global_selected_config = None
    temperature_selected_config = None
    if args.calibration_mode == "fixed":
        selected, records = select_fixed_config(group_stats)
    elif args.calibration_mode == "temperature":
        selected, records, temperature_selected_config = select_temperature_config(group_stats, temperature_grid)
    elif args.calibration_mode == "global":
        selected, records, global_selected_config = select_global_config(group_stats, eval_configs)
    else:
        selected, records = select_group_configs(
            group_stats,
            eval_configs if args.calibration_mode == "group_no_safety" else candidate_configs,
            args.min_group_queries,
            args.safe_delta,
            lock_missing_text=args.lock_missing_text,
            enforce_safety=args.calibration_mode in ("hpsac", "alpha_only"),
        )
    effective_level1_configs = build_effective_level1_configs(
        group_stats,
        selected,
        lock_missing_text=args.lock_missing_text,
    )
    non_fallback_count = count_non_fallback_effective(effective_level1_configs)
    non_fallback_both_have_text_count = count_non_fallback_both_have_text(effective_level1_configs)
    hpsac_stats = build_hpsac_stats(
        records,
        effective_level1_configs,
        args.min_group_queries,
        args.safe_delta,
    )
    print(
        "HPSAC selected groups | level1_groups=%d | non_fallback_effective=%d | non_fallback_both_have_text=%d"
        % (len(effective_level1_configs), non_fallback_count, non_fallback_both_have_text_count)
    )
    print(
        "HPSAC fallback stats | all_levels_total=%d | all_levels_specific=%d | all_levels_fallback=%d | effective_level1_specific=%d | effective_level1_fallback=%d"
        % (
            hpsac_stats["all_levels_total_group_count"],
            hpsac_stats["all_levels_accepted_specific_parameter_group_count"],
            hpsac_stats["all_levels_fallback_default_group_count"],
            hpsac_stats["effective_level1_policy"]["specific_parameter_group_count"],
            hpsac_stats["effective_level1_policy"]["default_fallback_group_count"],
        )
    )

    valid_eval = evaluate_policy_and_fallback(
        model_a,
        model_b,
        calibration_valid_triples,
        hr_to_tails,
        tr_to_heads,
        has_text,
        selected,
        use_gpu,
        subset_eval=True,
        split_name="calibration_valid" if args.separate_calibration_split else "valid",
        lock_missing_text=args.lock_missing_text,
    )
    test_eval = evaluate_policy_and_fallback(
        model_a,
        model_b,
        test_triples,
        hr_to_tails,
        tr_to_heads,
        has_text,
        selected,
        use_gpu,
        subset_eval=args.subset_eval,
        split_name="test",
        lock_missing_text=args.lock_missing_text,
    )

    print_metrics("HPSAC validation overall:", valid_eval["selected_overall"])
    if valid_eval["selected_subset"] is not None:
        print_metrics(
            "HPSAC validation Head/Tail Missing Text:",
            valid_eval["selected_subset"]["head_or_tail_missing_text"],
        )
        print_metrics(
            "HPSAC validation Head Missing Text:",
            valid_eval["selected_subset"]["head_missing_text"],
        )
        print_metrics(
            "HPSAC validation Tail Missing Text:",
            valid_eval["selected_subset"]["tail_missing_text"],
        )
        print_metrics(
            "HPSAC validation Head/Tail Both Have Text:",
            valid_eval["selected_subset"]["head_and_tail_have_text"],
        )
    print_metrics("HPSAC test overall:", test_eval["selected_overall"])
    if test_eval["selected_subset"] is not None:
        print_metrics(
            "HPSAC test Head/Tail Missing Text:",
            test_eval["selected_subset"]["head_or_tail_missing_text"],
        )
        print_metrics(
            "HPSAC test Head Missing Text:",
            test_eval["selected_subset"]["head_missing_text"],
        )
        print_metrics(
            "HPSAC test Tail Missing Text:",
            test_eval["selected_subset"]["tail_missing_text"],
        )
        print_metrics(
            "HPSAC test Head/Tail Both Have Text:",
            test_eval["selected_subset"]["head_and_tail_have_text"],
        )
    valid_missing_text_equal_bv1 = missing_text_matches_bv1(
        valid_eval["selected_subset"],
        valid_eval["bv1_subset"],
    )
    test_missing_text_equal_bv1 = missing_text_matches_bv1(
        test_eval["selected_subset"],
        test_eval["bv1_subset"],
    )
    test_overall_delta = metric_delta(test_eval["selected_overall"], test_eval["bv1_overall"])
    test_both_have_text_delta = subset_delta(
        test_eval["selected_subset"],
        test_eval["bv1_subset"],
        "head_and_tail_have_text",
    )
    print(
        "MT-Guard summary | lock_missing_text=%s | valid_missing_text_equal_bv1=%s | test_missing_text_equal_bv1=%s | test_overall_mrr_delta=%.6f | test_both_have_text_mrr_delta=%s | non_fallback_both_have_text=%d"
        % (
            "True" if args.lock_missing_text else "False",
            "NA" if valid_missing_text_equal_bv1 is None else str(valid_missing_text_equal_bv1),
            "NA" if test_missing_text_equal_bv1 is None else str(test_missing_text_equal_bv1),
            test_overall_delta["mrr"],
            (
                "NA"
                if test_both_have_text_delta is None else "%.6f" % test_both_have_text_delta["mrr"]
            ),
            non_fallback_both_have_text_count,
        )
    )
    eval_wall_time_sec = time.perf_counter() - run_start_time

    result_payload = {
        "dataset": args.dataset,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "requested_lambda_grid": requested_lambda_grid,
        "requested_alpha_grid": alpha_grid,
        "lambda_grid": lambda_grid,
        "alpha_grid": effective_alpha_grid,
        "temperature_grid": temperature_grid,
        "fallback_lambda": FALLBACK_LAMBDA,
        "fallback_alpha": FALLBACK_ALPHA,
        "fallback_temperature": 1.0,
        "lock_missing_text": bool(args.lock_missing_text),
        "calibration_mode": args.calibration_mode,
        "calibration_search_space": {
            "fixed_default_parameters": args.calibration_mode == "fixed",
            "group_level_search_enabled": args.calibration_mode in ("hpsac", "group_no_safety", "alpha_only"),
            "lambda_search_enabled": args.calibration_mode in ("hpsac", "group_no_safety", "global"),
            "alpha_search_enabled": args.calibration_mode in ("hpsac", "group_no_safety", "global", "alpha_only"),
            "temperature_search_enabled": args.calibration_mode == "temperature",
            "temperature_scaling_rank_invariant_for_positive_T": args.calibration_mode == "temperature",
        },
        "separate_calibration_split": bool(args.separate_calibration_split),
        "validation_split_seed": int(args.validation_split_seed),
        "validation_split_info": validation_split_info,
        "validation_metrics_scope": (
            "hpsac_calibration_half" if args.separate_calibration_split else "full_validation_legacy"
        ),
        "min_group_queries": args.min_group_queries,
        "safe_delta": args.safe_delta,
        "retrieval_topk": args.retrieval_topk,
        "retrieval_pool_size": args.retrieval_pool_size,
        "retrieval_source": args.retrieval_source,
        "inject_text_missing_rate": args.inject_text_missing_rate,
        "text_missing_mask_strategy": args.text_missing_mask_strategy,
        "text_missing_mask_seed": args.text_missing_mask_seed,
        "text_missing_mask_path": os.path.abspath(args.text_missing_mask_path) if args.text_missing_mask_path is not None else None,
        "save_text_missing_mask_path": os.path.abspath(args.save_text_missing_mask_path) if args.save_text_missing_mask_path is not None else None,
        "injection_info": injection_info,
        "injected_text_count": int(injected_text_mask.sum().item()) if injected_text_mask is not None else 0,
        "simulate_native_text_missing_rate": args.simulate_native_text_missing_rate,
        "simulated_native_text_missing_info": simulated_native_text_info,
        "simulated_native_text_missing_count": int(simulated_native_text_mask.sum().item()) if simulated_native_text_mask is not None else 0,
        "missing_text_count": int((~has_text).sum().item()),
        "runtime_cost": {
            "eval_wall_time_sec": float(eval_wall_time_sec),
        },
        "hpsac_stats": hpsac_stats,
        "group_records": serialize_group_records(records),
        "global_selected_config": global_selected_config,
        "temperature_selected_config": temperature_selected_config,
        "effective_level1_group_configs": effective_level1_configs,
        "level1_group_count": len(effective_level1_configs),
        "non_fallback_effective_group_count": int(non_fallback_count),
        "non_fallback_both_have_text_group_count": int(non_fallback_both_have_text_count),
        "validation_missing_text_equal_bv1": valid_missing_text_equal_bv1,
        "test_missing_text_equal_bv1": test_missing_text_equal_bv1,
        "validation_overall_metrics": valid_eval["selected_overall"],
        "validation_subset_metrics": valid_eval["selected_subset"],
        "hpsac_calibration_overall_metrics": valid_eval["selected_overall"],
        "hpsac_calibration_subset_metrics": valid_eval["selected_subset"],
        "validation_bv1_overall_metrics": valid_eval["bv1_overall"],
        "validation_bv1_subset_metrics": valid_eval["bv1_subset"],
        "validation_delta_vs_bv1": {
            "overall": metric_delta(valid_eval["selected_overall"], valid_eval["bv1_overall"]),
            "head_or_tail_missing_text": subset_delta(
                valid_eval["selected_subset"],
                valid_eval["bv1_subset"],
                "head_or_tail_missing_text",
            ),
            "head_missing_text": subset_delta(
                valid_eval["selected_subset"],
                valid_eval["bv1_subset"],
                "head_missing_text",
            ),
            "tail_missing_text": subset_delta(
                valid_eval["selected_subset"],
                valid_eval["bv1_subset"],
                "tail_missing_text",
            ),
            "head_and_tail_have_text": subset_delta(
                valid_eval["selected_subset"],
                valid_eval["bv1_subset"],
                "head_and_tail_have_text",
            ),
        },
        "test_overall_metrics": test_eval["selected_overall"],
        "test_subset_metrics": test_eval["selected_subset"],
        "test_bv1_overall_metrics": test_eval["bv1_overall"],
        "test_bv1_subset_metrics": test_eval["bv1_subset"],
        "test_delta_vs_bv1": {
            "overall": metric_delta(test_eval["selected_overall"], test_eval["bv1_overall"]),
            "head_or_tail_missing_text": subset_delta(
                test_eval["selected_subset"],
                test_eval["bv1_subset"],
                "head_or_tail_missing_text",
            ),
            "head_missing_text": subset_delta(
                test_eval["selected_subset"],
                test_eval["bv1_subset"],
                "head_missing_text",
            ),
            "tail_missing_text": subset_delta(
                test_eval["selected_subset"],
                test_eval["bv1_subset"],
                "tail_missing_text",
            ),
            "head_and_tail_have_text": subset_delta(
                test_eval["selected_subset"],
                test_eval["bv1_subset"],
                "head_and_tail_have_text",
            ),
        },
        "validation_policy_counts": valid_eval["policy_counts"],
        "test_policy_counts": test_eval["policy_counts"],
        "entity_count": ent_tot,
        "relation_count": rel_tot,
        "candidate_order": "entity_id_ascending_0_to_ent_tot_minus_1",
        "checkpoint_id_order_check": "state_dict entity/relation embedding shapes match dataset counts",
        "disabled_training_side_modules": {
            "confidence_calibration": True,
            "relation_aware_retrieval": True,
            "gate_expert_router_imputer_training": True,
            "diffheg": True,
        },
    }
    print("RESULT_JSON: " + json.dumps(result_payload, sort_keys=True))


if __name__ == "__main__":
    main()
