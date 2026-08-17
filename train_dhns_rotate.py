import argparse
import ctypes
import json
import os
import random
import time

import mmkgc
import numpy as np
import torch
from tqdm import tqdm

from mmkgc.config import Trainer_dhns, Tester_dhns
from mmkgc.module.model import AdvMixRotatE
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.strategy import NegativeSampling_complex
from mmkgc.data import TrainDataLoader_complex, TestDataLoader_complex
from mmkgc.adv.mmmodules_rotate import DiffHEG
from missing_text_protocol import (
    apply_simulated_native_text_missing as protocol_apply_simulated_native_text_missing,
    apply_text_missing_injection as protocol_apply_text_missing_injection,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate DHNS-RotatE on a selected MMKG dataset.")
    parser.add_argument("--dataset", type=str, default="MKG-Y")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--text-mode",
        type=str,
        choices=["soft_token", "zero_padding", "text_off"],
        default=None,
        help="Unified text ablation switch. soft_token uses the learnable missing-text token; zero_padding keeps missing text as a fixed zero projected vector; text_off strictly disables the text branch.",
    )
    parser.add_argument(
        "--inject-text-missing-rate",
        type=float,
        default=0.0,
        help="Additional text-missing rate injected over entities that originally have text. Example: 0.1 masks 10%% of currently available text entities.",
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
        "--save-text-missing-mask-path",
        type=str,
        default=None,
        help="Optional path for saving the generated or loaded entity-level text-missing mask for reproducibility.",
    )
    parser.add_argument(
        "--simulate-native-text-missing-rate",
        type=float,
        default=0.0,
        help="Zero a fraction of originally available text embeddings before original_has_text is computed, making DB15K-style complete text look like native text missingness for controlled diagnostics.",
    )
    parser.add_argument(
        "--inject-image-missing-rate",
        type=float,
        default=0.0,
        help="Additional image-missing rate injected over entities that originally have image. Example: 0.3 masks 30%% of currently available image entities.",
    )
    parser.add_argument("--use-missing-mask", action="store_true")
    parser.add_argument("--use-soft-missing-text", action="store_true")
    parser.add_argument(
        "--use-fixed-zero-missing-text",
        action="store_true",
        help="For text-missing entities, replace the projected text branch with an exact zero vector instead of a learnable token.",
    )
    parser.add_argument(
        "--use-side-aware-missing-text",
        action="store_true",
        help="On top of the A soft missing-text baseline, use head/tail-specific missing-text tokens on query scoring paths while keeping the shared token path for generator-side inputs.",
    )
    parser.add_argument("--use-soft-missing-image", action="store_true")
    parser.add_argument(
        "--use-prototype-missing-text",
        action="store_true",
        help="Build on the current soft missing-text baseline and replace the shared missing_text_token with prototype_token[cluster_id] for missing-text entities.",
    )
    parser.add_argument(
        "--prototype-cluster-path",
        type=str,
        default=None,
        help="Optional path to a prototype missing-text cluster file. Defaults to benchmarks/<dataset>/prototype_missing_text_clusters.pt.",
    )
    parser.add_argument(
        "--use-structure-conditioned-missing-text",
        action="store_true",
        help="Replace the shared soft missing-text token on truly missing-text entities with a structural-embedding-conditioned proxy text embedding.",
    )
    parser.add_argument(
        "--use-soft-token-text-generator-alignment",
        action="store_true",
        help="Align DiffHEG text inputs with the current soft missing-text token path during training/sampling.",
    )
    parser.add_argument("--use-missing-aware-fusion", action="store_true")
    parser.add_argument("--missing-text-attention-scale", type=float, default=1.0)
    parser.add_argument("--debug-fusion-sanity", action="store_true")
    parser.add_argument("--use-missingness-relation-expert", action="store_true")
    parser.add_argument("--expert-hidden-dim", type=int, default=128)
    parser.add_argument("--expert-num", type=int, default=2)
    parser.add_argument("--use-branch-local-relation-gate", action="store_true")
    parser.add_argument("--branch-gate-hidden-dim", type=int, default=64)
    parser.add_argument("--use-learnable-missing-text-gate", action="store_true")
    parser.add_argument("--use-oracle-restore-injected-text", action="store_true")
    parser.add_argument("--use-missing-aware-joint-scoring", action="store_true")
    parser.add_argument("--use-masked-fixed-denominator-joint-scoring", action="store_true")
    parser.add_argument("--use-availability-router", action="store_true")
    parser.add_argument(
        "--availability-router-mode",
        type=str,
        choices=["query_masked_softmax"],
        default="query_masked_softmax",
        help="Availability-only score router used at the final score-combination stage.",
    )
    parser.add_argument(
        "--availability-router-eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for availability-only routing normalization.",
    )
    parser.add_argument("--use-missing-aware-conditioning", action="store_true")
    parser.add_argument("--use-missing-aware-film-conditioning", action="store_true")
    parser.add_argument("--use-learned-reliability-conditioning", action="store_true")
    parser.add_argument("--debug-joint-scoring-sanity", action="store_true")
    parser.add_argument("--debug-missing-aware-joint-scoring", action="store_true")
    parser.add_argument("--debug-missing-aware-joint-scoring-batches", type=int, default=3)
    parser.add_argument("--subset-eval", action="store_true")
    parser.add_argument("--debug-masking", action="store_true")
    parser.add_argument("--debug-mask-batches", type=int, default=3)
    parser.add_argument("--debug-reliability", action="store_true")
    parser.add_argument("--debug-reliability-batches", type=int, default=3)
    parser.add_argument("--use-text-loss-gating", action="store_true")
    parser.add_argument("--use-text-sampling-gating", action="store_true")
    parser.add_argument("--use-missing-text-aux-loss", action="store_true")
    parser.add_argument("--missing-text-aux-weight", type=float, default=0.1)
    parser.add_argument("--pseudo-missing-prob", type=float, default=0.0)
    parser.add_argument("--missing-sample-weight", type=float, default=1.0)
    parser.add_argument(
        "--use-entity-specific-missing-text",
        action="store_true",
        help="Replace the shared soft missing-text token with entity-specific surrogate text predicted from structural and visual signals.",
    )
    parser.add_argument(
        "--use-retrieval-missing-text",
        action="store_true",
        help="Augment the soft missing-text token with KNN prototype text aggregated from structurally similar entities.",
    )
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument(
        "--retrieval-pool-size",
        type=int,
        default=512,
        help="Lightweight retrieval candidate pool size. Set 0 to use all observed-text entities.",
    )
    parser.add_argument(
        "--retrieval-source",
        type=str,
        choices=["entity_embedding_knn", "random_text_pool"],
        default="entity_embedding_knn",
    )
    parser.add_argument("--retrieval-mix-weight", type=float, default=1.0)
    parser.add_argument("--use-retrieval-confidence-calibration", action="store_true")
    parser.add_argument(
        "--retrieval-confidence-type",
        type=str,
        choices=["mean_topk_similarity", "normalized_mean_topk_similarity"],
        default="mean_topk_similarity",
    )
    parser.add_argument("--retrieval-confidence-min", type=float, default=0.1)
    parser.add_argument("--retrieval-confidence-max", type=float, default=1.0)
    parser.add_argument("--use-cross-modal-text-imputer", action="store_true")
    parser.add_argument("--text-imputer-hidden-dim", type=int, default=256)
    parser.add_argument("--text-imputer-residual-weight", type=float, default=0.05)
    parser.add_argument("--text-imputer-rec-weight", type=float, default=0.01)
    parser.add_argument("--text-imputer-nce-weight", type=float, default=0.01)
    parser.add_argument("--text-imputer-temperature", type=float, default=0.07)
    parser.add_argument("--use-confidence-gated-retrieval", action="store_true")
    parser.add_argument(
        "--retrieval-gate-type",
        type=str,
        choices=["similarity_based"],
        default="similarity_based",
    )
    parser.add_argument("--retrieval-gate-min", type=float, default=0.0)
    parser.add_argument("--retrieval-gate-max", type=float, default=1.0)
    parser.add_argument("--use-relation-aware-retrieval", action="store_true")
    parser.add_argument("--min-relation-pool-size", type=int, default=32)
    parser.add_argument(
        "--relation-retrieval-fallback",
        type=str,
        choices=["global_text_pool"],
        default="global_text_pool",
    )
    parser.add_argument(
        "--entity-specific-missing-text-recon-weight",
        type=float,
        default=0.0,
        help="Auxiliary reconstruction weight for matching predicted surrogate text to observed text entities.",
    )
    parser.add_argument("--use-missing-text-consistency", action="store_true")
    parser.add_argument("--consistency-prob", type=float, default=0.1)
    parser.add_argument("--consistency-lambda", type=float, default=0.05)
    parser.add_argument("--use-missing-text-token-scale", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoint/rotate.ckpt")
    parser.add_argument("--test", action="store_true", help="Skip training, load --checkpoint-path, and run evaluation only.")
    parser.add_argument("--train-times", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--neg-ent", type=int, default=64)
    parser.add_argument(
        "--ns-strategy",
        type=str,
        choices=["dhns", "uniform", "bernoulli"],
        default="dhns",
        help="Negative sampling strategy. dhns preserves the existing DiffHEG/DHNS path; uniform/bernoulli train RotatE with normal entity corruption only.",
    )
    parser.add_argument(
        "--dhns-use-bernoulli-normal-sampling",
        action="store_true",
        help="For ns_strategy=dhns only, keep DHNS/DiffHEG enabled but use normal Bernoulli entity corruption for the base training sampler.",
    )
    parser.add_argument(
        "--bernoulli-use-cross-sampling",
        action="store_true",
        help="For ns_strategy=bernoulli only, train the Bernoulli baseline with the same cross head/tail sampler used by the DHNS baseline.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Use nbatches-based sampling. When set, batch size is derived as train_triples // num_batches.",
    )
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--lrg", type=float, default=0.002)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--g-epoch", type=int, default=10)
    parser.add_argument("--rotate-dim", type=int, default=512)
    parser.add_argument("--rotate-margin", type=float, default=6.0)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument(
        "--record-missing-token-diagnostics",
        action="store_true",
        help="Record per-epoch missing-text attention beta and missing-token gradient norms for reviewer diagnostics.",
    )
    parser.add_argument("--diagnostic-epoch-interval", type=int, default=1)
    parser.add_argument(
        "--diagnostic-entity-sample-size",
        type=int,
        default=2048,
        help="Sample up to this many missing-text and observed-text entities for attention diagnostics. Set 0 to use all entities.",
    )
    parser.add_argument("--diagnostic-batch-size", type=int, default=4096)
    parser.add_argument(
        "--fusion-probe-output-path",
        type=str,
        default=None,
        help="Optional .pt output containing final entity fusion representations z_e and attention weights for PCA/t-SNE.",
    )
    parser.add_argument(
        "--result-json-output-path",
        type=str,
        default=None,
        help="Optional path to save the final RESULT_JSON payload.",
    )
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        ctypes.cdll.msvcrt.srand(seed)
    except Exception:
        pass


def summarize_missingness(img_emb, text_emb):
    num_entities = text_emb.shape[0]
    if img_emb.shape[0] != num_entities:
        raise ValueError("Image/text embedding row counts do not match.")

    has_text = text_emb.float().norm(dim=1).ne(0)
    has_image = img_emb.float().norm(dim=1).ne(0)
    missing_text_count = int((~has_text).sum().item())
    missing_image_count = int((~has_image).sum().item())

    print("Embedding missingness summary:")
    print(f"  num_entities: {num_entities}")
    print(f"  missing_text_count: {missing_text_count} ({missing_text_count / num_entities:.4%})")
    print(f"  missing_image_count: {missing_image_count} ({missing_image_count / num_entities:.4%})")

    return has_text, has_image


def build_relation_text_candidate_pools(train_path, rel_tot, has_text):
    head_sets = [set() for _ in range(rel_tot)]
    tail_sets = [set() for _ in range(rel_tot)]
    has_text_cpu = has_text.detach().cpu().bool()

    with open(train_path, "r", encoding="utf-8") as fin:
        first_line = fin.readline()
        try:
            expected_count = int(first_line.strip())
        except ValueError:
            expected_count = None
            fin.seek(0)

        observed_count = 0
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            h, t, r = (int(parts[0]), int(parts[1]), int(parts[2]))
            if not 0 <= r < rel_tot:
                continue
            if 0 <= h < has_text_cpu.numel() and bool(has_text_cpu[h].item()):
                head_sets[r].add(h)
            if 0 <= t < has_text_cpu.numel() and bool(has_text_cpu[t].item()):
                tail_sets[r].add(t)
            observed_count += 1

    relation_head_pools = [torch.tensor(sorted(pool), dtype=torch.long) for pool in head_sets]
    relation_tail_pools = [torch.tensor(sorted(pool), dtype=torch.long) for pool in tail_sets]
    head_sizes = [pool.numel() for pool in relation_head_pools]
    tail_sizes = [pool.numel() for pool in relation_tail_pools]
    pool_info = {
        "train_path": train_path,
        "expected_train_triples": expected_count,
        "observed_train_triples": observed_count,
        "avg_head_pool_size": float(np.mean(head_sizes)) if head_sizes else 0.0,
        "avg_tail_pool_size": float(np.mean(tail_sizes)) if tail_sizes else 0.0,
        "min_head_pool_size": int(min(head_sizes)) if head_sizes else 0,
        "min_tail_pool_size": int(min(tail_sizes)) if tail_sizes else 0,
        "max_head_pool_size": int(max(head_sizes)) if head_sizes else 0,
        "max_tail_pool_size": int(max(tail_sizes)) if tail_sizes else 0,
    }
    return relation_head_pools, relation_tail_pools, pool_info


def install_relation_aware_negative_score_adapter(model):
    original_get_batch_rel_embs = model.get_batch_rel_embs
    original_mm_negative_score = model.mm_negative_score
    relation_id_attr = "_dhns_relation_ids"

    def get_batch_rel_embs_with_ids(data):
        relation_embs = original_get_batch_rel_embs(data)
        # Keep relation ids attached to their own embedding tensor; no model-global "last batch" cache.
        try:
            setattr(relation_embs, relation_id_attr, data.detach())
        except Exception as error:
            raise RuntimeError("Failed to attach explicit relation ids for relation-aware retrieval.") from error
        if getattr(relation_embs, relation_id_attr, None) is None:
            raise RuntimeError("Explicit relation ids were not attached for relation-aware retrieval.")
        return relation_embs

    def mm_negative_score_with_relation_ids(*args, **kwargs):
        if kwargs.get("batch_r_ids") is None:
            batch_r = kwargs.get("batch_r")
            if batch_r is None and len(args) >= 2:
                batch_r = args[1]
            relation_ids = getattr(batch_r, relation_id_attr, None)
            if relation_ids is not None:
                kwargs["batch_r_ids"] = relation_ids
        return original_mm_negative_score(*args, **kwargs)

    model.get_batch_rel_embs = get_batch_rel_embs_with_ids
    model.mm_negative_score = mm_negative_score_with_relation_ids


def apply_text_missing_injection(
    text_emb,
    inject_rate,
    seed,
    mask_strategy="random",
    benchmark_path=None,
    mask_path=None,
    save_mask_path=None,
):
    return protocol_apply_text_missing_injection(
        text_emb,
        inject_rate,
        seed=seed,
        mask_strategy=mask_strategy,
        benchmark_path=benchmark_path,
        mask_path=mask_path,
        save_mask_path=save_mask_path,
    )


def apply_simulated_native_text_missing(text_emb, missing_rate, seed):
    return protocol_apply_simulated_native_text_missing(text_emb, missing_rate, seed=seed)


def apply_image_missing_injection(img_emb, inject_rate, seed):
    if inject_rate <= 0:
        return img_emb, None, None
    if inject_rate > 1:
        raise ValueError("--inject-image-missing-rate must be in [0, 1].")

    injected_img_emb = img_emb.clone()
    injected_image_mask = torch.zeros(img_emb.shape[0], dtype=torch.bool)
    original_has_image = img_emb.float().norm(dim=1).ne(0)
    available_indices = torch.nonzero(original_has_image, as_tuple=False).view(-1)
    available_count = int(available_indices.numel())
    inject_count = int(round(available_count * inject_rate))

    if inject_count <= 0:
        info = {
            "inject_rate_requested": float(inject_rate),
            "inject_rate_applied": 0.0,
            "available_image_before": available_count,
            "additional_masked_count": 0,
        }
        return injected_img_emb, info, injected_image_mask

    rng = np.random.default_rng(seed)
    selected = rng.choice(available_count, size=inject_count, replace=False)
    inject_indices = available_indices[torch.as_tensor(selected, dtype=torch.long)]
    injected_img_emb[inject_indices] = 0
    injected_image_mask[inject_indices] = True

    info = {
        "inject_rate_requested": float(inject_rate),
        "inject_rate_applied": inject_count / available_count if available_count > 0 else 0.0,
        "available_image_before": available_count,
        "additional_masked_count": inject_count,
    }
    return injected_img_emb, info, injected_image_mask


def print_overall_metrics(metrics):
    print("Overall link prediction metrics:")
    print(f"  MRR: {metrics['mrr']:.6f}")
    print(f"  MR: {metrics['mr']:.6f}")
    print(f"  hit@10: {metrics['hit10']:.6f}")
    print(f"  hit@3: {metrics['hit3']:.6f}")
    print(f"  hit@1: {metrics['hit1']:.6f}")


def print_subset_metrics(subset_metrics):
    if subset_metrics is None:
        return
    print("Modality-availability subset metrics:")
    for group_name, metrics in subset_metrics.items():
        print(
            "  %s | query_count=%d | triple_count=%d | MRR=%.6f | MR=%.6f | hit@10=%.6f | hit@3=%.6f | hit@1=%.6f"
            % (
                group_name,
                metrics["count"],
                metrics["triple_count"],
                metrics["mrr"],
                metrics["mr"],
                metrics["hit10"],
                metrics["hit3"],
                metrics["hit1"],
            )
        )


def print_subset_sanity(sanity):
    if sanity is None:
        return
    print("Subset evaluation sanity check:")
    print(f"  count_meaning: {sanity['count_meaning']}")
    overall = sanity["overall_query_metrics_same_loop"]
    print(
        "  same_loop_overall | query_count=%d | triple_count=%d | MRR=%.6f | MR=%.6f | hit@10=%.6f | hit@3=%.6f | hit@1=%.6f"
        % (
            overall["count"],
            overall["triple_count"],
            overall["mrr"],
            overall["mr"],
            overall["hit10"],
            overall["hit3"],
            overall["hit1"],
        )
    )
    if "text_partition_recombined_metrics" in sanity:
        partition = sanity["text_partition_recombined_metrics"]
        print(
            "  text_partition_recombined(head_or_tail_missing_text + head_and_tail_have_text) | query_count=%d | triple_count=%d | MRR=%.6f | MR=%.6f | hit@10=%.6f | hit@3=%.6f | hit@1=%.6f"
            % (
                partition["count"],
                partition["triple_count"],
                partition["mrr"],
                partition["mr"],
                partition["hit10"],
                partition["hit3"],
                partition["hit1"],
            )
        )
    if "image_partition_recombined_metrics" in sanity:
        partition = sanity["image_partition_recombined_metrics"]
        print(
            "  image_partition_recombined(head_or_tail_missing_image + head_and_tail_have_image) | query_count=%d | triple_count=%d | MRR=%.6f | MR=%.6f | hit@10=%.6f | hit@3=%.6f | hit@1=%.6f"
            % (
                partition["count"],
                partition["triple_count"],
                partition["mrr"],
                partition["mr"],
                partition["hit10"],
                partition["hit3"],
                partition["hit1"],
            )
        )


def to_tensor(data, use_gpu):
    tensor = torch.from_numpy(data)
    return tensor.cuda() if use_gpu else tensor


def train_standard_negative_sampling(model, data_loader, train_times, alpha, use_gpu, strategy_label):
    if use_gpu:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    print(f"Finish initializing {strategy_label} RotatE trainer...")

    training_range = tqdm(range(train_times))
    for epoch in training_range:
        epoch_loss = 0.0
        for data in data_loader:
            optimizer.zero_grad()
            loss, _ = model({
                "batch_h": to_tensor(data["batch_h"], use_gpu),
                "batch_t": to_tensor(data["batch_t"], use_gpu),
                "batch_r": to_tensor(data["batch_r"], use_gpu),
                "batch_y": to_tensor(data["batch_y"], use_gpu),
                "mode": data["mode"],
            })
            auxiliary_loss = getattr(model.model, "consume_auxiliary_loss", lambda: None)()
            if auxiliary_loss is not None:
                loss += auxiliary_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        training_range.set_description("Epoch %d | KGC loss: %f" % (epoch, epoch_loss))


def resolve_text_mode(args):
    if args.text_mode is not None:
        return args.text_mode
    if (
        args.use_soft_missing_text or
        args.use_structure_conditioned_missing_text or
        args.use_retrieval_missing_text
    ):
        return "soft_token"
    if args.use_fixed_zero_missing_text:
        return "zero_padding"
    return "legacy"


def resolve_prototype_cluster_path(args):
    if args.prototype_cluster_path is not None:
        return args.prototype_cluster_path
    return os.path.join(".", "benchmarks", args.dataset, "prototype_missing_text_clusters.pt")


def load_prototype_missing_text_clusters(path, expected_ent_tot):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prototype cluster file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    metadata = {}
    if isinstance(payload, dict):
        if "cluster_ids" not in payload:
            raise KeyError("Prototype cluster file must contain 'cluster_ids'.")
        cluster_ids = payload["cluster_ids"]
        metadata = {k: v for k, v in payload.items() if k != "cluster_ids"}
    else:
        cluster_ids = payload

    cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long).view(-1).cpu()
    if cluster_ids.shape[0] != expected_ent_tot:
        raise ValueError(
            f"Prototype cluster count mismatch: expected {expected_ent_tot} entities, got {cluster_ids.shape[0]}."
        )
    if bool((cluster_ids < 0).any().item()):
        raise ValueError("Prototype cluster ids must be non-negative.")

    inferred_num_clusters = int(cluster_ids.max().item()) + 1 if cluster_ids.numel() > 0 else 0
    num_clusters = int(metadata.get("num_clusters", inferred_num_clusters))
    if num_clusters <= 0:
        raise ValueError("Prototype cluster file must define at least one cluster.")
    if inferred_num_clusters > num_clusters:
        raise ValueError(
            f"Prototype cluster ids require at least {inferred_num_clusters} clusters, but file declares {num_clusters}."
        )

    return cluster_ids, {
        "path": os.path.abspath(path),
        "num_clusters": num_clusters,
        "entity_count": int(cluster_ids.shape[0]),
        "unique_cluster_count": int(torch.unique(cluster_ids).numel()),
        "dataset": metadata.get("dataset"),
        "seed": metadata.get("seed"),
    }


def estimate_retrieval_index_info(has_text, relation_head_pools=None, relation_tail_pools=None):
    observed_text_count = int(has_text.sum().item()) if has_text is not None else 0
    relation_pool_entity_refs = 0
    if relation_head_pools is not None:
        relation_pool_entity_refs += sum(int(pool.numel()) for pool in relation_head_pools)
    if relation_tail_pools is not None:
        relation_pool_entity_refs += sum(int(pool.numel()) for pool in relation_tail_pools)
    id_bytes = 8
    return {
        "external_vector_index": False,
        "global_text_pool_entity_count": observed_text_count,
        "global_text_pool_index_size_mb": float(observed_text_count * id_bytes / (1024.0 * 1024.0)),
        "relation_pool_entity_reference_count": int(relation_pool_entity_refs),
        "relation_pool_index_size_mb": float(relation_pool_entity_refs * id_bytes / (1024.0 * 1024.0)),
        "index_note": "Retrieval reuses learned entity/text embeddings; stored index cost is candidate entity ids only.",
    }


def summarize_values(values):
    finite_values = [
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ]
    summary = {
        "count": int(len(values)),
        "finite_count": int(len(finite_values)),
        "mean": None,
        "std": None,
        "min": None,
        "p25": None,
        "median": None,
        "p75": None,
        "max": None,
    }
    if not finite_values:
        return summary
    arr = np.asarray(finite_values, dtype=np.float64)
    summary.update({
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    })
    return summary


def summarize_tensor(tensor):
    if tensor is None:
        return summarize_values([])
    return summarize_values(tensor.detach().view(-1).float().cpu().tolist())


def select_diagnostic_entity_ids(has_text, sample_size, seed):
    has_text = has_text.detach().cpu().bool().view(-1)
    missing_ids = torch.nonzero(~has_text, as_tuple=False).view(-1)
    observed_ids = torch.nonzero(has_text, as_tuple=False).view(-1)

    def _sample(ids, seed_offset):
        if sample_size <= 0 or ids.numel() <= sample_size:
            return ids
        generator = torch.Generator()
        generator.manual_seed(int(seed) + seed_offset)
        order = torch.randperm(ids.numel(), generator=generator)[:sample_size]
        return ids.index_select(0, order)

    return torch.unique(
        torch.cat([_sample(missing_ids, 1009), _sample(observed_ids, 2003)], dim=0),
        sorted=True,
    )


def summarize_fusion_probe(probe):
    z_e = probe["z_e"].float()
    has_text = probe["has_text"].bool()
    modalities = probe.get("modalities", [])
    text_index = modalities.index("text") if "text" in modalities else None
    beta_text = probe["attention"][:, text_index].float() if text_index is not None else None

    def _group(mask):
        count = int(mask.sum().item())
        if count == 0:
            return {
                "count": 0,
                "z_norm": summarize_values([]),
                "z_centroid_norm": None,
                "z_mean_abs": None,
                "text_norm": summarize_values([]),
                "raw_text_norm": summarize_values([]),
                "text_beta": summarize_values([]),
            }
        group_z = z_e[mask]
        result = {
            "count": count,
            "z_norm": summarize_tensor(group_z.norm(dim=-1)),
            "z_centroid_norm": float(group_z.mean(dim=0).norm().item()),
            "z_mean_abs": float(group_z.abs().mean().item()),
            "text_norm": summarize_tensor(probe["text"][mask].norm(dim=-1)) if "text" in probe else summarize_values([]),
            "raw_text_norm": summarize_tensor(probe["raw_text"][mask].norm(dim=-1)) if "raw_text" in probe else summarize_values([]),
            "text_beta": summarize_tensor(beta_text[mask]) if beta_text is not None else summarize_values([]),
        }
        return result

    missing_mask = ~has_text
    observed_mask = has_text
    return {
        "entity_count": int(z_e.shape[0]),
        "modalities": modalities,
        "missing_text": _group(missing_mask),
        "observed_text": _group(observed_mask),
        "all": _group(torch.ones_like(has_text, dtype=torch.bool)),
    }


def save_fusion_probe(model, has_text, args, resolved_text_mode):
    if args.fusion_probe_output_path is None:
        return None, None
    entity_ids = select_diagnostic_entity_ids(
        has_text,
        sample_size=args.diagnostic_entity_sample_size,
        seed=args.seed,
    )
    probe = model.export_fusion_probe(entity_ids, batch_size=args.diagnostic_batch_size)
    probe["metadata"] = {
        "model": "AdvMixRotatE",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "text_mode": resolved_text_mode,
        "use_missing_mask": bool(args.use_missing_mask),
        "use_soft_missing_text": bool(resolved_text_mode == "soft_token"),
        "use_fixed_zero_missing_text": bool(args.use_fixed_zero_missing_text or resolved_text_mode == "zero_padding"),
        "inject_text_missing_rate": float(args.inject_text_missing_rate),
        "text_missing_mask_strategy": args.text_missing_mask_strategy,
        "text_missing_mask_path": os.path.abspath(args.text_missing_mask_path) if args.text_missing_mask_path is not None else None,
    }
    output_path = os.path.abspath(args.fusion_probe_output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(probe, output_path)
    summary = summarize_fusion_probe(probe)
    print(
        "Fusion probe saved | path=%s | entities=%d | missing_text=%d | observed_text=%d"
        % (
            output_path,
            summary["entity_count"],
            summary["missing_text"]["count"],
            summary["observed_text"]["count"],
        )
    )
    return output_path, summary


def main():
    run_start_time = time.perf_counter()
    args = parse_args()
    if not 0.0 <= args.pseudo_missing_prob <= 1.0:
        raise ValueError("--pseudo-missing-prob must be in [0, 1].")
    if not 0.0 <= args.inject_text_missing_rate <= 1.0:
        raise ValueError("--inject-text-missing-rate must be in [0, 1].")
    if args.text_missing_mask_path is not None and not os.path.exists(args.text_missing_mask_path):
        raise FileNotFoundError(f"Text-missing mask file not found: {args.text_missing_mask_path}")
    if args.missing_sample_weight < 1.0:
        raise ValueError("--missing-sample-weight must be >= 1.0.")
    if args.missing_text_aux_weight < 0.0:
        raise ValueError("--missing-text-aux-weight must be >= 0.0.")
    if args.availability_router_eps <= 0.0:
        raise ValueError("--availability-router-eps must be > 0.")
    if not 0.0 <= args.inject_image_missing_rate <= 1.0:
        raise ValueError("--inject-image-missing-rate must be in [0, 1].")
    if not 0.0 <= args.consistency_prob <= 1.0:
        raise ValueError("--consistency-prob must be in [0, 1].")
    if args.consistency_lambda < 0.0:
        raise ValueError("--consistency-lambda must be >= 0.0.")
    if args.diagnostic_epoch_interval <= 0:
        raise ValueError("--diagnostic-epoch-interval must be > 0.")
    if args.diagnostic_entity_sample_size < 0:
        raise ValueError("--diagnostic-entity-sample-size must be >= 0.")
    if args.diagnostic_batch_size <= 0:
        raise ValueError("--diagnostic-batch-size must be > 0.")
    if args.expert_hidden_dim <= 0:
        raise ValueError("--expert-hidden-dim must be > 0.")
    if args.expert_num <= 0:
        raise ValueError("--expert-num must be > 0.")
    if args.branch_gate_hidden_dim <= 0:
        raise ValueError("--branch-gate-hidden-dim must be > 0.")
    if args.use_structure_conditioned_missing_text and args.use_entity_specific_missing_text:
        raise ValueError("--use-structure-conditioned-missing-text cannot be combined with --use-entity-specific-missing-text.")
    if args.use_retrieval_missing_text and args.use_entity_specific_missing_text:
        raise ValueError("--use-retrieval-missing-text cannot be combined with --use-entity-specific-missing-text.")
    if args.use_retrieval_missing_text and args.use_structure_conditioned_missing_text:
        raise ValueError("--use-retrieval-missing-text cannot be combined with --use-structure-conditioned-missing-text.")
    if args.use_retrieval_missing_text and args.use_prototype_missing_text:
        raise ValueError("--use-retrieval-missing-text cannot be combined with the old --use-prototype-missing-text path.")
    if args.retrieval_topk <= 0:
        raise ValueError("--retrieval-topk must be > 0.")
    if args.retrieval_pool_size < 0:
        raise ValueError("--retrieval-pool-size must be >= 0.")
    if args.retrieval_mix_weight < 0.0:
        raise ValueError("--retrieval-mix-weight must be >= 0.0.")
    if not 0.0 <= args.retrieval_confidence_min <= 1.0:
        raise ValueError("--retrieval-confidence-min must be in [0, 1].")
    if not 0.0 <= args.retrieval_confidence_max <= 1.0:
        raise ValueError("--retrieval-confidence-max must be in [0, 1].")
    if args.retrieval_confidence_min > args.retrieval_confidence_max:
        raise ValueError("--retrieval-confidence-min must be <= --retrieval-confidence-max.")
    if args.text_imputer_hidden_dim <= 0:
        raise ValueError("--text-imputer-hidden-dim must be > 0.")
    if args.text_imputer_residual_weight < 0.0:
        raise ValueError("--text-imputer-residual-weight must be >= 0.0.")
    if args.text_imputer_rec_weight < 0.0:
        raise ValueError("--text-imputer-rec-weight must be >= 0.0.")
    if args.text_imputer_nce_weight < 0.0:
        raise ValueError("--text-imputer-nce-weight must be >= 0.0.")
    if args.text_imputer_temperature <= 0.0:
        raise ValueError("--text-imputer-temperature must be > 0.")
    if args.use_cross_modal_text_imputer and not args.use_retrieval_missing_text:
        raise ValueError("--use-cross-modal-text-imputer requires --use-retrieval-missing-text.")
    if args.use_cross_modal_text_imputer and args.use_retrieval_confidence_calibration:
        raise ValueError("--use-cross-modal-text-imputer should not be combined with confidence calibration.")
    if args.use_cross_modal_text_imputer and args.use_confidence_gated_retrieval:
        raise ValueError("--use-cross-modal-text-imputer should not be combined with confidence-gated retrieval.")
    if args.use_cross_modal_text_imputer and args.use_relation_aware_retrieval:
        raise ValueError("--use-cross-modal-text-imputer keeps B-v1 retrieval and should not use relation-aware retrieval.")
    if args.use_cross_modal_text_imputer and args.use_branch_local_relation_gate:
        raise ValueError("--use-cross-modal-text-imputer should not be combined with branch-local relation gate.")
    if args.use_cross_modal_text_imputer and args.use_missingness_relation_expert:
        raise ValueError("--use-cross-modal-text-imputer should not be combined with missingness-relation expert.")
    if args.use_cross_modal_text_imputer and args.use_missing_text_aux_loss:
        raise ValueError("--use-cross-modal-text-imputer should not be combined with missing-text aux loss.")
    if args.use_cross_modal_text_imputer and args.use_missing_text_consistency:
        raise ValueError("--use-cross-modal-text-imputer should not be combined with missing-text consistency loss.")
    if not 0.0 <= args.retrieval_gate_min <= 1.0:
        raise ValueError("--retrieval-gate-min must be in [0, 1].")
    if not 0.0 <= args.retrieval_gate_max <= 1.0:
        raise ValueError("--retrieval-gate-max must be in [0, 1].")
    if args.retrieval_gate_min > args.retrieval_gate_max:
        raise ValueError("--retrieval-gate-min must be <= --retrieval-gate-max.")
    if args.use_confidence_gated_retrieval and not args.use_retrieval_missing_text:
        raise ValueError("--use-confidence-gated-retrieval requires --use-retrieval-missing-text.")
    if args.use_retrieval_confidence_calibration and not args.use_retrieval_missing_text:
        raise ValueError("--use-retrieval-confidence-calibration requires --use-retrieval-missing-text.")
    if args.use_retrieval_confidence_calibration and args.use_confidence_gated_retrieval:
        raise ValueError("--use-retrieval-confidence-calibration cannot be combined with --use-confidence-gated-retrieval.")
    if args.use_relation_aware_retrieval and not args.use_retrieval_missing_text:
        raise ValueError("--use-relation-aware-retrieval requires --use-retrieval-missing-text.")
    if args.min_relation_pool_size <= 0:
        raise ValueError("--min-relation-pool-size must be > 0.")
    if args.use_prototype_missing_text and args.use_structure_conditioned_missing_text:
        raise ValueError("--use-prototype-missing-text is the minimal B module on top of the A baseline and cannot be combined with --use-structure-conditioned-missing-text.")
    if args.use_prototype_missing_text and args.use_entity_specific_missing_text:
        raise ValueError("--use-prototype-missing-text is the minimal B module on top of the A baseline and cannot be combined with --use-entity-specific-missing-text.")
    set_seed(args.seed)
    resolved_text_mode = resolve_text_mode(args)
    disable_text = resolved_text_mode == "text_off"
    effective_use_soft_missing_text = (
        args.use_soft_missing_text or
        args.use_structure_conditioned_missing_text or
        args.use_retrieval_missing_text
    )
    effective_use_fixed_zero_missing_text = args.use_fixed_zero_missing_text or resolved_text_mode == "zero_padding"
    if args.text_mode is not None:
        effective_use_soft_missing_text = resolved_text_mode == "soft_token"
    if effective_use_soft_missing_text and effective_use_fixed_zero_missing_text:
        raise ValueError("Soft missing-text token and fixed zero missing-text cannot be enabled together.")
    if disable_text and effective_use_fixed_zero_missing_text:
        raise ValueError("--text-mode text_off cannot be combined with fixed zero missing-text.")
    if effective_use_fixed_zero_missing_text and (
        args.use_side_aware_missing_text or
        args.use_prototype_missing_text or
        args.use_structure_conditioned_missing_text or
        args.use_entity_specific_missing_text or
        args.use_retrieval_missing_text
    ):
        raise ValueError("Fixed zero missing-text is a baseline and cannot be combined with soft-token replacement variants.")
    if args.use_prototype_missing_text and not effective_use_soft_missing_text:
        raise ValueError("--use-prototype-missing-text must be used together with the A baseline via --use-soft-missing-text or --text-mode soft_token.")
    if args.use_side_aware_missing_text and not effective_use_soft_missing_text:
        raise ValueError("--use-side-aware-missing-text must be used together with the A baseline via --use-soft-missing-text or --text-mode soft_token.")
    if disable_text and args.use_structure_conditioned_missing_text:
        raise ValueError("--use-structure-conditioned-missing-text requires the text branch to be enabled.")
    if disable_text and args.use_retrieval_missing_text:
        raise ValueError("--use-retrieval-missing-text requires the text branch to be enabled.")
    if disable_text and args.use_retrieval_confidence_calibration:
        raise ValueError("--use-retrieval-confidence-calibration requires the text branch to be enabled.")
    if disable_text and args.use_cross_modal_text_imputer:
        raise ValueError("--use-cross-modal-text-imputer requires the text branch to be enabled.")
    if disable_text and args.use_missing_text_aux_loss:
        raise ValueError("--use-missing-text-aux-loss requires the text branch to be enabled.")
    if disable_text and args.use_missingness_relation_expert:
        raise ValueError("--use-missingness-relation-expert requires the text branch to be enabled.")
    if args.use_missingness_relation_expert and args.expert_num != 2:
        raise ValueError("--use-missingness-relation-expert currently supports only --expert-num 2.")
    if disable_text and args.use_branch_local_relation_gate:
        raise ValueError("--use-branch-local-relation-gate requires the text branch to be enabled.")
    if args.use_branch_local_relation_gate and not args.use_retrieval_missing_text:
        raise ValueError("--use-branch-local-relation-gate requires --use-retrieval-missing-text.")
    if args.use_missing_text_consistency and not effective_use_soft_missing_text:
        raise ValueError("--use-missing-text-consistency requires the soft missing-text baseline to be enabled.")
    if args.use_missing_text_consistency and args.use_structure_conditioned_missing_text:
        raise ValueError("--use-missing-text-consistency is only supported for the minimal soft missing-text token baseline.")
    if args.use_missing_text_consistency and args.use_entity_specific_missing_text:
        raise ValueError("--use-missing-text-consistency is only supported for the minimal soft missing-text token baseline.")
    if args.use_missing_text_consistency and args.use_prototype_missing_text:
        raise ValueError("--use-missing-text-consistency is only supported for the minimal A baseline and cannot be combined with --use-prototype-missing-text.")
    if args.use_missing_text_token_scale and not effective_use_soft_missing_text:
        raise ValueError("--use-missing-text-token-scale requires the soft missing-text baseline to be enabled.")
    if args.use_missing_text_token_scale and args.use_structure_conditioned_missing_text:
        raise ValueError("--use-missing-text-token-scale is only supported for the minimal soft missing-text token baseline.")
    if args.use_missing_text_token_scale and args.use_entity_specific_missing_text:
        raise ValueError("--use-missing-text-token-scale is only supported for the minimal soft missing-text token baseline.")
    if args.use_missing_text_token_scale and args.use_prototype_missing_text:
        raise ValueError("--use-missing-text-token-scale is only supported for the minimal A baseline and cannot be combined with --use-prototype-missing-text.")
    if args.use_missing_text_token_scale and args.use_missing_text_consistency:
        raise ValueError("--use-missing-text-token-scale is intended for the minimal A+B calibration setting and cannot be combined with consistency.")

    if args.dhns_use_bernoulli_normal_sampling and args.ns_strategy != "dhns":
        raise ValueError("--dhns-use-bernoulli-normal-sampling can only be used with --ns-strategy dhns.")
    if args.bernoulli_use_cross_sampling and args.ns_strategy != "bernoulli":
        raise ValueError("--bernoulli-use-cross-sampling can only be used with --ns-strategy bernoulli.")

    use_gpu = torch.cuda.is_available() and not args.no_gpu
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    dhns_bernoulli_normal_sampling = args.ns_strategy == "dhns" and args.dhns_use_bernoulli_normal_sampling
    bernoulli_cross_sampling = args.ns_strategy == "bernoulli" and args.bernoulli_use_cross_sampling
    train_sampling_mode = (
        "normal"
        if (args.ns_strategy in ("uniform", "bernoulli") and not bernoulli_cross_sampling)
        or dhns_bernoulli_normal_sampling
        else "cross"
    )
    train_bern_flag = 1 if (args.ns_strategy == "bernoulli" and not bernoulli_cross_sampling) or dhns_bernoulli_normal_sampling else 0
    benchmark_path = f"./benchmarks/{args.dataset}/"
    visual_path = f"./embeddings/{args.dataset}-visual.pth"
    textual_path = f"./embeddings/{args.dataset}-textual.pth"

    print(
        f"Experiment config | model=AdvMixRotatE | dataset={args.dataset} | seed={args.seed} "
        f"| ns_strategy={args.ns_strategy} | train_sampling_mode={train_sampling_mode} | bern_flag={train_bern_flag} "
        f"| dhns_use_bernoulli_normal_sampling={args.dhns_use_bernoulli_normal_sampling} "
        f"| bernoulli_use_cross_sampling={args.bernoulli_use_cross_sampling} "
        f"| text_mode={resolved_text_mode} "
        f"| use_missing_mask={args.use_missing_mask} | use_text_loss_gating={args.use_text_loss_gating} "
        f"| use_text_sampling_gating={args.use_text_sampling_gating} | use_soft_missing_text={effective_use_soft_missing_text} "
        f"| use_fixed_zero_missing_text={effective_use_fixed_zero_missing_text} "
        f"| use_side_aware_missing_text={args.use_side_aware_missing_text} "
        f"| use_soft_missing_image={args.use_soft_missing_image} "
        f"| use_prototype_missing_text={args.use_prototype_missing_text} "
        f"| use_structure_conditioned_missing_text={args.use_structure_conditioned_missing_text} "
        f"| use_soft_token_text_generator_alignment={args.use_soft_token_text_generator_alignment} "
        f"| use_missing_aware_fusion={args.use_missing_aware_fusion} | missing_text_attention_scale={args.missing_text_attention_scale} "
        f"| use_missingness_relation_expert={args.use_missingness_relation_expert} "
        f"| expert_hidden_dim={args.expert_hidden_dim} "
        f"| expert_num={args.expert_num} "
        f"| use_branch_local_relation_gate={args.use_branch_local_relation_gate} "
        f"| branch_gate_hidden_dim={args.branch_gate_hidden_dim} "
        f"| debug_fusion_sanity={args.debug_fusion_sanity} | use_learnable_missing_text_gate={args.use_learnable_missing_text_gate} "
        f"| use_oracle_restore_injected_text={args.use_oracle_restore_injected_text} "
        f"| test_only={args.test} "
        f"| use_missing_aware_joint_scoring={args.use_missing_aware_joint_scoring} "
        f"| use_masked_fixed_denominator_joint_scoring={args.use_masked_fixed_denominator_joint_scoring} "
        f"| use_availability_router={args.use_availability_router} "
        f"| availability_router_mode={args.availability_router_mode} "
        f"| availability_router_eps={args.availability_router_eps} "
        f"| use_missing_aware_conditioning={args.use_missing_aware_conditioning} "
        f"| use_missing_aware_film_conditioning={args.use_missing_aware_film_conditioning} "
        f"| use_learned_reliability_conditioning={args.use_learned_reliability_conditioning} "
        f"| debug_joint_scoring_sanity={args.debug_joint_scoring_sanity} "
        f"| debug_missing_aware_joint_scoring={args.debug_missing_aware_joint_scoring} "
        f"| debug_missing_aware_joint_scoring_batches={args.debug_missing_aware_joint_scoring_batches} "
        f"| debug_reliability={args.debug_reliability} | debug_reliability_batches={args.debug_reliability_batches} "
        f"| inject_text_missing_rate={args.inject_text_missing_rate} "
        f"| text_missing_mask_strategy={args.text_missing_mask_strategy} "
        f"| text_missing_mask_path={args.text_missing_mask_path} "
        f"| save_text_missing_mask_path={args.save_text_missing_mask_path} "
        f"| simulate_native_text_missing_rate={args.simulate_native_text_missing_rate} "
        f"| inject_image_missing_rate={args.inject_image_missing_rate} "
        f"| use_missing_text_aux_loss={args.use_missing_text_aux_loss} "
        f"| missing_text_aux_weight={args.missing_text_aux_weight} "
        f"| pseudo_missing_prob={args.pseudo_missing_prob} "
        f"| missing_sample_weight={args.missing_sample_weight} "
        f"| use_entity_specific_missing_text={args.use_entity_specific_missing_text} "
        f"| use_retrieval_missing_text={args.use_retrieval_missing_text} "
        f"| retrieval_topk={args.retrieval_topk} "
        f"| retrieval_pool_size={args.retrieval_pool_size} "
        f"| retrieval_source={args.retrieval_source} "
        f"| retrieval_mix_weight={args.retrieval_mix_weight} "
        f"| use_retrieval_confidence_calibration={args.use_retrieval_confidence_calibration} "
        f"| retrieval_confidence_type={args.retrieval_confidence_type} "
        f"| retrieval_confidence_min={args.retrieval_confidence_min} "
        f"| retrieval_confidence_max={args.retrieval_confidence_max} "
        f"| use_cross_modal_text_imputer={args.use_cross_modal_text_imputer} "
        f"| text_imputer_hidden_dim={args.text_imputer_hidden_dim} "
        f"| text_imputer_residual_weight={args.text_imputer_residual_weight} "
        f"| text_imputer_rec_weight={args.text_imputer_rec_weight} "
        f"| text_imputer_nce_weight={args.text_imputer_nce_weight} "
        f"| text_imputer_temperature={args.text_imputer_temperature} "
        f"| use_confidence_gated_retrieval={args.use_confidence_gated_retrieval} "
        f"| retrieval_gate_type={args.retrieval_gate_type} "
        f"| retrieval_gate_min={args.retrieval_gate_min} "
        f"| retrieval_gate_max={args.retrieval_gate_max} "
        f"| use_relation_aware_retrieval={args.use_relation_aware_retrieval} "
        f"| min_relation_pool_size={args.min_relation_pool_size} "
        f"| relation_retrieval_fallback={args.relation_retrieval_fallback} "
        f"| entity_specific_missing_text_recon_weight={args.entity_specific_missing_text_recon_weight} "
        f"| use_missing_text_consistency={args.use_missing_text_consistency} "
        f"| consistency_prob={args.consistency_prob} "
        f"| consistency_lambda={args.consistency_lambda} "
        f"| use_missing_text_token_scale={args.use_missing_text_token_scale} "
        f"| record_missing_token_diagnostics={args.record_missing_token_diagnostics} "
        f"| diagnostic_epoch_interval={args.diagnostic_epoch_interval} "
        f"| diagnostic_entity_sample_size={args.diagnostic_entity_sample_size} "
        f"| diagnostic_batch_size={args.diagnostic_batch_size} "
        f"| fusion_probe_output_path={args.fusion_probe_output_path} "
        f"| result_json_output_path={args.result_json_output_path}"
    )

    train_batch_size = None if args.num_batches is not None else args.batch_size
    train_dataloader = TrainDataLoader_complex(
        in_path=benchmark_path,
        batch_size=train_batch_size,
        nbatches=args.num_batches,
        threads=args.threads,
        sampling_mode=train_sampling_mode,
        bern_flag=train_bern_flag,
        filter_flag=1,
        neg_ent=args.neg_ent,
        neg_rel=0,
    )

    test_dataloader = TestDataLoader_complex(benchmark_path, "link")

    original_img_emb = torch.load(visual_path)
    original_text_emb = torch.load(textual_path)
    original_text_emb, simulated_native_text_info, simulated_native_text_mask = apply_simulated_native_text_missing(
        original_text_emb,
        args.simulate_native_text_missing_rate,
        args.seed,
    )
    img_emb, image_injection_info, injected_image_mask = apply_image_missing_injection(
        original_img_emb, args.inject_image_missing_rate, args.seed
    )
    original_has_text = original_text_emb.float().norm(dim=1).ne(0)
    text_emb, injection_info, injected_text_mask = apply_text_missing_injection(
        original_text_emb,
        args.inject_text_missing_rate,
        args.seed,
        mask_strategy=args.text_missing_mask_strategy,
        benchmark_path=benchmark_path,
        mask_path=args.text_missing_mask_path,
        save_mask_path=args.save_text_missing_mask_path,
    )
    if simulated_native_text_info is not None:
        print("Simulated native text missingness summary:")
        print(f"  requested_native_missing_rate: {simulated_native_text_info['simulate_rate_requested']:.4%}")
        print(f"  applied_native_missing_rate: {simulated_native_text_info['simulate_rate_applied']:.4%}")
        print(f"  available_text_before_simulation: {simulated_native_text_info['available_text_before']}")
        print(f"  simulated_native_missing_text_entities: {simulated_native_text_info['simulated_native_missing_count']}")
    if image_injection_info is not None:
        print("Injected image missingness summary:")
        print(f"  requested_additional_missing_rate: {image_injection_info['inject_rate_requested']:.4%}")
        print(f"  applied_additional_missing_rate: {image_injection_info['inject_rate_applied']:.4%}")
        print(f"  available_image_before_injection: {image_injection_info['available_image_before']}")
        print(f"  additionally_masked_image_entities: {image_injection_info['additional_masked_count']}")
    if injection_info is not None:
        print("Injected text missingness summary:")
        print(f"  requested_additional_missing_rate: {injection_info['inject_rate_requested']:.4%}")
        print(f"  applied_additional_missing_rate: {injection_info['inject_rate_applied']:.4%}")
        print(f"  available_text_before_injection: {injection_info['available_text_before']}")
        print(f"  additionally_masked_text_entities: {injection_info['additional_masked_count']}")
        print(f"  mask_strategy: {injection_info['mask_strategy']}")
        print(f"  mask_source: {injection_info['mask_source']}")
        print(f"  mask_checksum_sha256: {injection_info['mask_checksum_sha256']}")
        if injection_info.get("save_mask_path") is not None:
            print(f"  saved_mask_path: {injection_info['save_mask_path']}")
    has_text, has_image = summarize_missingness(img_emb, text_emb)
    if args.use_structure_conditioned_missing_text:
        print("Structure-conditioned missing-text summary:")
        print(f"  truly_missing_text_count: {int((~original_has_text).sum().item())}")
        print(f"  structure_conditioned_proxy_enabled: {args.use_structure_conditioned_missing_text}")

    prototype_cluster_ids = None
    prototype_cluster_info = None
    if args.use_prototype_missing_text:
        prototype_cluster_path = resolve_prototype_cluster_path(args)
        prototype_cluster_ids, prototype_cluster_info = load_prototype_missing_text_clusters(
            prototype_cluster_path,
            expected_ent_tot=train_dataloader.get_ent_tot(),
        )
        print("Prototype missing-text summary:")
        print(f"  cluster_file: {prototype_cluster_info['path']}")
        print(f"  num_clusters: {prototype_cluster_info['num_clusters']}")
        print(f"  unique_cluster_count: {prototype_cluster_info['unique_cluster_count']}")
        print(f"  entity_count: {prototype_cluster_info['entity_count']}")

    relation_head_text_pools = None
    relation_tail_text_pools = None
    relation_pool_info = None
    if args.use_relation_aware_retrieval:
        train_path = os.path.join(benchmark_path, "train2id.txt")
        relation_head_text_pools, relation_tail_text_pools, relation_pool_info = build_relation_text_candidate_pools(
            train_path,
            train_dataloader.get_rel_tot(),
            has_text,
        )
        print("Relation-aware retrieval pool summary:")
        print(f"  train_path: {relation_pool_info['train_path']}")
        print(f"  observed_train_triples: {relation_pool_info['observed_train_triples']}")
        print(f"  avg_head_pool_size: {relation_pool_info['avg_head_pool_size']:.2f}")
        print(f"  avg_tail_pool_size: {relation_pool_info['avg_tail_pool_size']:.2f}")
        print(f"  min_relation_pool_size: {args.min_relation_pool_size}")
        print(f"  fallback: {args.relation_retrieval_fallback}")

    rotate_dim = args.rotate_dim
    rotate_margin = args.rotate_margin
    rotate_epsilon = 2.0
    print(
        "Resolved training hyperparameters | rotate_dim=%d | rotate_margin=%.6f | "
        "rotate_epsilon=%.6f | train_times=%d | batch_size=%d | num_batches=%d | "
        "neg_ent=%d | alpha=%.8f | lrg=%.8f | mu=%.6f | g_epoch=%d"
        % (
            rotate_dim,
            rotate_margin,
            rotate_epsilon,
            args.train_times,
            train_dataloader.get_batch_size(),
            len(train_dataloader),
            args.neg_ent,
            args.alpha,
            args.lrg,
            args.mu,
            args.g_epoch,
        )
    )

    rotate = AdvMixRotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=rotate_dim,
        margin=rotate_margin,
        epsilon=rotate_epsilon,
        img_emb=img_emb,
        text_emb=text_emb,
        has_text=has_text,
        original_has_text=original_has_text,
        has_image=has_image,
        use_missing_mask=args.use_missing_mask,
        use_soft_missing_text=effective_use_soft_missing_text,
        use_fixed_zero_missing_text=effective_use_fixed_zero_missing_text,
        use_side_aware_missing_text=args.use_side_aware_missing_text,
        use_soft_missing_image=args.use_soft_missing_image,
        use_prototype_missing_text=args.use_prototype_missing_text,
        prototype_missing_text_cluster_ids=prototype_cluster_ids,
        prototype_missing_text_num_clusters=(
            prototype_cluster_info["num_clusters"] if prototype_cluster_info is not None else None
        ),
        use_structure_conditioned_missing_text=args.use_structure_conditioned_missing_text,
        use_missing_aware_fusion=args.use_missing_aware_fusion,
        missing_text_attention_scale=args.missing_text_attention_scale,
        debug_fusion_sanity=args.debug_fusion_sanity,
        use_learnable_missing_text_gate=args.use_learnable_missing_text_gate,
        use_oracle_restore_injected_text=args.use_oracle_restore_injected_text,
        oracle_text_emb=original_text_emb,
        injected_text_mask=injected_text_mask,
        use_missing_aware_joint_scoring=args.use_missing_aware_joint_scoring,
        use_masked_fixed_denominator_joint_scoring=args.use_masked_fixed_denominator_joint_scoring,
        use_availability_router=args.use_availability_router,
        availability_router_mode=args.availability_router_mode,
        availability_router_eps=args.availability_router_eps,
        debug_joint_scoring_sanity=args.debug_joint_scoring_sanity,
        debug_missing_aware_joint_scoring=args.debug_missing_aware_joint_scoring,
        debug_missing_aware_joint_scoring_batches=args.debug_missing_aware_joint_scoring_batches,
        disable_text=disable_text,
        pseudo_missing_prob=args.pseudo_missing_prob,
        use_soft_token_text_generator_alignment=args.use_soft_token_text_generator_alignment,
        use_missingness_relation_expert=args.use_missingness_relation_expert,
        expert_hidden_dim=args.expert_hidden_dim,
        expert_num=args.expert_num,
        use_branch_local_relation_gate=args.use_branch_local_relation_gate,
        branch_gate_hidden_dim=args.branch_gate_hidden_dim,
        use_entity_specific_missing_text=args.use_entity_specific_missing_text,
        use_retrieval_missing_text=args.use_retrieval_missing_text,
        retrieval_topk=args.retrieval_topk,
        retrieval_pool_size=args.retrieval_pool_size,
        retrieval_source=args.retrieval_source,
        retrieval_mix_weight=args.retrieval_mix_weight,
        use_retrieval_confidence_calibration=args.use_retrieval_confidence_calibration,
        retrieval_confidence_type=args.retrieval_confidence_type,
        retrieval_confidence_min=args.retrieval_confidence_min,
        retrieval_confidence_max=args.retrieval_confidence_max,
        use_cross_modal_text_imputer=args.use_cross_modal_text_imputer,
        text_imputer_hidden_dim=args.text_imputer_hidden_dim,
        text_imputer_residual_weight=args.text_imputer_residual_weight,
        text_imputer_rec_weight=args.text_imputer_rec_weight,
        text_imputer_nce_weight=args.text_imputer_nce_weight,
        text_imputer_temperature=args.text_imputer_temperature,
        use_confidence_gated_retrieval=args.use_confidence_gated_retrieval,
        retrieval_gate_type=args.retrieval_gate_type,
        retrieval_gate_min=args.retrieval_gate_min,
        retrieval_gate_max=args.retrieval_gate_max,
        use_relation_aware_retrieval=args.use_relation_aware_retrieval,
        relation_head_text_pools=relation_head_text_pools,
        relation_tail_text_pools=relation_tail_text_pools,
        min_relation_pool_size=args.min_relation_pool_size,
        relation_retrieval_fallback=args.relation_retrieval_fallback,
        entity_specific_missing_text_recon_weight=args.entity_specific_missing_text_recon_weight,
        use_missing_text_consistency=args.use_missing_text_consistency,
        consistency_prob=args.consistency_prob,
        consistency_lambda=args.consistency_lambda,
        use_missing_text_token_scale=args.use_missing_text_token_scale,
    )
    if args.use_relation_aware_retrieval:
        install_relation_aware_negative_score_adapter(rotate)
    active_modalities = rotate.get_active_modalities()
    print(
        "Resolved text configuration | text_mode=%s | active_modalities=%s | modality_count=%d"
        % (resolved_text_mode, ",".join(active_modalities), len(active_modalities))
    )

    train_wall_time_sec = 0.0
    missing_token_diagnostics_state = {
        "enabled": bool(args.record_missing_token_diagnostics),
        "diagnostic_epoch_interval": int(args.diagnostic_epoch_interval),
        "diagnostic_entity_sample_size": int(args.diagnostic_entity_sample_size),
        "diagnostic_batch_size": int(args.diagnostic_batch_size),
        "diagnostic_seed": int(args.seed),
        "history": [],
        "last": None,
    }
    if args.test:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for --test mode: {args.checkpoint_path}")
        print(f"Test-only mode: loading checkpoint from {args.checkpoint_path}")
        missing_text_aux_state = {
            "enabled": bool(args.use_missing_text_aux_loss),
            "missing_text_aux_weight": args.missing_text_aux_weight,
            "current_batch_missing_text_count": 0,
            "aux_loss_value": 0.0,
        }
    else:
        if use_gpu:
            torch.cuda.synchronize()
        train_start_time = time.perf_counter()
        model = NegativeSampling_complex(
            model=rotate,
            loss=SigmoidLoss(adv_temperature=0.5),
            batch_size=train_dataloader.get_batch_size(),
            l3_regul_rate=0.000005,
        )

        if args.ns_strategy in ("uniform", "bernoulli"):
            strategy_label = "Bernoulli-Cross" if args.bernoulli_use_cross_sampling else (
                "Bernoulli" if args.ns_strategy == "bernoulli" else "Uniform"
            )
            train_standard_negative_sampling(
                model=model,
                data_loader=train_dataloader,
                train_times=args.train_times,
                alpha=args.alpha,
                use_gpu=use_gpu,
                strategy_label=strategy_label,
            )
            missing_text_aux_state = {
                "enabled": False,
                "missing_text_aux_weight": args.missing_text_aux_weight,
                "current_batch_missing_text_count": 0,
                "aux_loss_value": 0.0,
            }
        else:
            adv_generator = DiffHEG(
                embedding_dim=rotate_dim * 2,
                T=50,
                dim_r=rotate_dim,
                margin=rotate_margin,
                eps=rotate_epsilon,
                use_missing_aware_conditioning=args.use_missing_aware_conditioning,
                use_missing_aware_film_conditioning=args.use_missing_aware_film_conditioning,
                availability_dim=rotate.get_conditioning_availability_dim(),
            )

            trainer = Trainer_dhns(
                model=model,
                data_loader=train_dataloader,
                train_times=args.train_times,
                alpha=args.alpha,
                use_gpu=use_gpu,
                opt_method="adam",
                generator=adv_generator,
                lrg=args.lrg,
                mu=args.mu,
                g_epoch=args.g_epoch,
                debug_masking=args.debug_masking,
                debug_mask_batches=args.debug_mask_batches,
                debug_reliability=args.debug_reliability,
                debug_reliability_batches=args.debug_reliability_batches,
                use_learned_reliability_conditioning=args.use_learned_reliability_conditioning,
                use_text_loss_gating=args.use_text_loss_gating,
                use_text_sampling_gating=args.use_text_sampling_gating,
                missing_sample_weight=args.missing_sample_weight,
                use_missing_text_aux_loss=args.use_missing_text_aux_loss,
                missing_text_aux_weight=args.missing_text_aux_weight,
                record_missing_token_diagnostics=args.record_missing_token_diagnostics,
                diagnostic_epoch_interval=args.diagnostic_epoch_interval,
                diagnostic_entity_sample_size=args.diagnostic_entity_sample_size,
                diagnostic_batch_size=args.diagnostic_batch_size,
                diagnostic_seed=args.seed,
            )
            trainer.run()
            missing_text_aux_state = trainer.get_missing_text_aux_state()
            missing_token_diagnostics_state = trainer.get_missing_token_diagnostics_state()
        if args.use_learnable_missing_text_gate:
            print(f"Learned missing-text gate: {rotate.get_missing_text_gate().item():.6f}")
        if args.use_missing_text_token_scale:
            print(f"Learned missing-text token scale: {rotate.missing_text_token_scale.item():.6f}")

        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        rotate.save_checkpoint(args.checkpoint_path)
        if use_gpu:
            torch.cuda.synchronize()
        train_wall_time_sec = time.perf_counter() - train_start_time

    if use_gpu:
        torch.cuda.synchronize()
    eval_start_time = time.perf_counter()
    rotate.load_checkpoint(args.checkpoint_path)
    tester = Tester_dhns(model=rotate, data_loader=test_dataloader, use_gpu=use_gpu)
    test_results = tester.run_link_prediction(type_constrain=False, subset_eval=args.subset_eval)
    if use_gpu:
        torch.cuda.synchronize()
    eval_wall_time_sec = time.perf_counter() - eval_start_time
    if args.subset_eval:
        overall_metrics, subset_metrics, subset_sanity = test_results
    else:
        overall_metrics = test_results
        subset_metrics = None
        subset_sanity = None

    print_overall_metrics(overall_metrics)
    print_subset_metrics(subset_metrics)
    print_subset_sanity(subset_sanity)
    branch_gate_state = rotate.get_branch_local_relation_gate_state()
    retrieval_confidence_state = rotate.get_retrieval_confidence_calibration_state()
    retrieval_confidence_stats = retrieval_confidence_state["last_stats"] or {}
    text_imputer_state = rotate.get_cross_modal_text_imputer_state()
    text_imputer_stats = text_imputer_state["last_stats"] or {}
    checkpoint_size_mb = (
        float(os.path.getsize(args.checkpoint_path) / (1024.0 * 1024.0))
        if os.path.exists(args.checkpoint_path) else None
    )
    gpu_memory_cost = {
        "gpu_enabled": bool(use_gpu),
        "max_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if use_gpu else None
        ),
        "max_memory_reserved_mb": (
            float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)) if use_gpu else None
        ),
    }
    runtime_cost = {
        "train_wall_time_sec": float(train_wall_time_sec),
        "eval_wall_time_sec": float(eval_wall_time_sec),
        "total_wall_time_sec": float(time.perf_counter() - run_start_time),
    }
    storage_cost = {
        "checkpoint_size_mb": checkpoint_size_mb,
        "retrieval_index_info": estimate_retrieval_index_info(
            has_text,
            relation_head_pools=relation_head_text_pools,
            relation_tail_pools=relation_tail_text_pools,
        ),
    }
    fusion_probe_output_path, fusion_probe_summary = save_fusion_probe(
        rotate,
        has_text,
        args,
        resolved_text_mode,
    )

    result_payload = {
        "model": "AdvMixRotatE",
        "dataset": args.dataset,
        "seed": args.seed,
        "test_only": args.test,
        "ns_strategy": args.ns_strategy,
        "train_sampling_mode": train_sampling_mode,
        "bern_flag": train_bern_flag,
        "dhns_use_bernoulli_normal_sampling": args.dhns_use_bernoulli_normal_sampling,
        "bernoulli_use_cross_sampling": args.bernoulli_use_cross_sampling,
        "bernoulli_relation_prior_enabled": bool(args.ns_strategy == "bernoulli" and train_bern_flag == 1),
        "dhns_generator_enabled": args.ns_strategy == "dhns",
        "rotate_dim": rotate_dim,
        "rotate_margin": rotate_margin,
        "rotate_epsilon": rotate_epsilon,
        "train_times": args.train_times,
        "configured_batch_size": args.batch_size,
        "configured_num_batches": args.num_batches,
        "effective_batch_size": train_dataloader.get_batch_size(),
        "effective_num_batches": len(train_dataloader),
        "neg_ent": args.neg_ent,
        "neg_rel": 0,
        "alpha": args.alpha,
        "lrg": args.lrg,
        "mu": args.mu,
        "g_epoch": args.g_epoch,
        "text_mode": resolved_text_mode,
        "inject_text_missing_rate": args.inject_text_missing_rate,
        "text_missing_mask_strategy": args.text_missing_mask_strategy,
        "text_missing_mask_path": os.path.abspath(args.text_missing_mask_path) if args.text_missing_mask_path is not None else None,
        "save_text_missing_mask_path": os.path.abspath(args.save_text_missing_mask_path) if args.save_text_missing_mask_path is not None else None,
        "simulate_native_text_missing_rate": args.simulate_native_text_missing_rate,
        "inject_image_missing_rate": args.inject_image_missing_rate,
        "use_missing_mask": args.use_missing_mask,
        "use_soft_missing_text": effective_use_soft_missing_text,
        "use_fixed_zero_missing_text": effective_use_fixed_zero_missing_text,
        "use_side_aware_missing_text": args.use_side_aware_missing_text,
        "use_soft_missing_image": args.use_soft_missing_image,
        "use_prototype_missing_text": args.use_prototype_missing_text,
        "prototype_cluster_path": prototype_cluster_info["path"] if prototype_cluster_info is not None else None,
        "prototype_num_clusters": prototype_cluster_info["num_clusters"] if prototype_cluster_info is not None else None,
        "prototype_unique_cluster_count": prototype_cluster_info["unique_cluster_count"] if prototype_cluster_info is not None else None,
        "use_structure_conditioned_missing_text": args.use_structure_conditioned_missing_text,
        "use_missing_aware_fusion": args.use_missing_aware_fusion,
        "missing_text_attention_scale": args.missing_text_attention_scale,
        "use_missingness_relation_expert": args.use_missingness_relation_expert,
        "expert_hidden_dim": args.expert_hidden_dim,
        "expert_num": args.expert_num,
        "use_branch_local_relation_gate": args.use_branch_local_relation_gate,
        "branch_gate_hidden_dim": args.branch_gate_hidden_dim,
        "missing_text_gate_mean": branch_gate_state["last_stats"].get("missing_text_gate_mean") if branch_gate_state["last_stats"] is not None else None,
        "complete_text_gate_mean": branch_gate_state["last_stats"].get("complete_text_gate_mean") if branch_gate_state["last_stats"] is not None else None,
        "gate_min": branch_gate_state["last_stats"].get("gate_min") if branch_gate_state["last_stats"] is not None else None,
        "gate_max": branch_gate_state["last_stats"].get("gate_max") if branch_gate_state["last_stats"] is not None else None,
        "branch_local_relation_gate_state": branch_gate_state,
        "debug_fusion_sanity": args.debug_fusion_sanity,
        "use_learnable_missing_text_gate": args.use_learnable_missing_text_gate,
        "learned_missing_text_gate": rotate.get_missing_text_gate().item() if args.use_learnable_missing_text_gate else None,
        "use_oracle_restore_injected_text": args.use_oracle_restore_injected_text,
        "use_missing_aware_joint_scoring": args.use_missing_aware_joint_scoring,
        "use_masked_fixed_denominator_joint_scoring": args.use_masked_fixed_denominator_joint_scoring,
        "use_availability_router": args.use_availability_router,
        "availability_router_mode": args.availability_router_mode,
        "availability_router_eps": args.availability_router_eps,
        "use_missing_aware_conditioning": args.use_missing_aware_conditioning,
        "use_missing_aware_film_conditioning": args.use_missing_aware_film_conditioning,
        "use_learned_reliability_conditioning": args.use_learned_reliability_conditioning,
        "debug_joint_scoring_sanity": args.debug_joint_scoring_sanity,
        "debug_missing_aware_joint_scoring": args.debug_missing_aware_joint_scoring,
        "debug_missing_aware_joint_scoring_batches": args.debug_missing_aware_joint_scoring_batches,
        "debug_reliability": args.debug_reliability,
        "debug_reliability_batches": args.debug_reliability_batches,
        "use_text_loss_gating": args.use_text_loss_gating,
        "use_text_sampling_gating": args.use_text_sampling_gating,
        "use_missing_text_aux_loss": args.use_missing_text_aux_loss,
        "missing_text_aux_weight": args.missing_text_aux_weight,
        "current_batch_missing_text_count": missing_text_aux_state["current_batch_missing_text_count"],
        "aux_loss_value": missing_text_aux_state["aux_loss_value"],
        "missing_text_aux_state": missing_text_aux_state,
        "use_soft_token_text_generator_alignment": args.use_soft_token_text_generator_alignment,
        "pseudo_missing_prob": args.pseudo_missing_prob,
        "missing_sample_weight": args.missing_sample_weight,
        "use_entity_specific_missing_text": args.use_entity_specific_missing_text,
        "use_retrieval_missing_text": args.use_retrieval_missing_text,
        "retrieval_topk": args.retrieval_topk,
        "retrieval_pool_size": args.retrieval_pool_size,
        "retrieval_source": args.retrieval_source,
        "retrieval_mix_weight": args.retrieval_mix_weight,
        "use_retrieval_confidence_calibration": args.use_retrieval_confidence_calibration,
        "retrieval_confidence_type": args.retrieval_confidence_type,
        "retrieval_confidence_min_config": args.retrieval_confidence_min,
        "retrieval_confidence_max_config": args.retrieval_confidence_max,
        "retrieval_confidence_mean": retrieval_confidence_stats.get("retrieval_confidence_mean"),
        "retrieval_confidence_min": retrieval_confidence_stats.get("retrieval_confidence_min"),
        "retrieval_confidence_max": retrieval_confidence_stats.get("retrieval_confidence_max"),
        "calibrated_retrieval_norm_mean": retrieval_confidence_stats.get("calibrated_retrieval_norm_mean"),
        "retrieval_confidence_calibration_state": retrieval_confidence_state,
        "use_cross_modal_text_imputer": args.use_cross_modal_text_imputer,
        "text_imputer_hidden_dim": args.text_imputer_hidden_dim,
        "text_imputer_residual_weight": args.text_imputer_residual_weight,
        "text_imputer_rec_weight": args.text_imputer_rec_weight,
        "text_imputer_nce_weight": args.text_imputer_nce_weight,
        "text_imputer_temperature": args.text_imputer_temperature,
        "text_imputer_rec_loss": text_imputer_stats.get("text_imputer_rec_loss"),
        "text_imputer_nce_loss": text_imputer_stats.get("text_imputer_nce_loss"),
        "text_imputer_residual_norm_mean": text_imputer_stats.get("text_imputer_residual_norm_mean"),
        "pseudo_text_norm_mean": text_imputer_stats.get("pseudo_text_norm_mean"),
        "prototype_text_agg_mean_norm": text_imputer_stats.get("prototype_text_agg_mean_norm"),
        "missing_text_compensated_norm_mean": text_imputer_stats.get("missing_text_compensated_norm_mean"),
        "cross_modal_text_imputer_state": text_imputer_state,
        "use_confidence_gated_retrieval": args.use_confidence_gated_retrieval,
        "retrieval_gate_type": args.retrieval_gate_type,
        "retrieval_gate_min": args.retrieval_gate_min,
        "retrieval_gate_max": args.retrieval_gate_max,
        "use_relation_aware_retrieval": args.use_relation_aware_retrieval,
        "min_relation_pool_size": args.min_relation_pool_size,
        "relation_retrieval_fallback": args.relation_retrieval_fallback,
        "relation_pool_info": relation_pool_info,
        "relation_aware_retrieval_stats": rotate.get_relation_aware_retrieval_stats(),
        "retrieval_missing_text_stats": rotate.get_retrieval_missing_text_stats(),
        "missingness_relation_expert_state": rotate.get_missingness_relation_expert_state(),
        "entity_specific_missing_text_recon_weight": args.entity_specific_missing_text_recon_weight,
        "use_missing_text_consistency": args.use_missing_text_consistency,
        "consistency_prob": args.consistency_prob,
        "consistency_lambda": args.consistency_lambda,
        "use_missing_text_token_scale": args.use_missing_text_token_scale,
        "learned_missing_text_token_scale": rotate.missing_text_token_scale.item() if args.use_missing_text_token_scale else None,
        "record_missing_token_diagnostics": args.record_missing_token_diagnostics,
        "diagnostic_epoch_interval": args.diagnostic_epoch_interval,
        "diagnostic_entity_sample_size": args.diagnostic_entity_sample_size,
        "diagnostic_batch_size": args.diagnostic_batch_size,
        "missing_token_diagnostics": missing_token_diagnostics_state,
        "fusion_probe_output_path": fusion_probe_output_path,
        "fusion_probe_summary": fusion_probe_summary,
        "result_json_output_path": os.path.abspath(args.result_json_output_path) if args.result_json_output_path is not None else None,
        "injection_info": injection_info,
        "simulated_native_text_missing_info": simulated_native_text_info,
        "image_injection_info": image_injection_info,
        "injected_text_count": int(injected_text_mask.sum().item()) if injected_text_mask is not None else 0,
        "simulated_native_text_missing_count": int(simulated_native_text_mask.sum().item()) if simulated_native_text_mask is not None else 0,
        "injected_image_count": int(injected_image_mask.sum().item()) if injected_image_mask is not None else 0,
        "truly_missing_text_count": int((~original_has_text).sum().item()),
        "missing_text_count": int((~has_text).sum().item()),
        "missing_image_count": int((~has_image).sum().item()),
        "active_modalities": active_modalities,
        "active_modality_count": len(active_modalities),
        "text_branch_debug": rotate.get_text_branch_debug_state(),
        "overall_metrics": overall_metrics,
        "subset_metrics": subset_metrics,
        "subset_sanity": subset_sanity,
        "runtime_cost": runtime_cost,
        "gpu_memory_cost": gpu_memory_cost,
        "storage_cost": storage_cost,
        "checkpoint_path": args.checkpoint_path,
    }
    result_json = json.dumps(result_payload, sort_keys=True)
    if args.result_json_output_path is not None:
        result_json_output_path = os.path.abspath(args.result_json_output_path)
        result_json_output_dir = os.path.dirname(result_json_output_path)
        if result_json_output_dir:
            os.makedirs(result_json_output_dir, exist_ok=True)
        with open(result_json_output_path, "w", encoding="utf-8") as fout:
            fout.write(result_json + "\n")
        print(f"RESULT_JSON saved to: {result_json_output_path}")
    print("RESULT_JSON: " + result_json)


if __name__ == "__main__":
    main()
