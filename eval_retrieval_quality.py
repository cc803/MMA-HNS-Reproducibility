import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from missing_text_protocol import apply_text_missing_injection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate whether retrieval neighbors for missing-text entities are structurally reasonable."
    )
    parser.add_argument("--dataset", type=str, default="MKG-Y")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--retrieval-source",
        type=str,
        choices=["both", "all", "entity_embedding_knn", "random_text_pool", "relation_constrained_knn"],
        default="both",
    )
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=512)
    parser.add_argument(
        "--relation-min-pool-size",
        type=int,
        default=32,
        help="Minimum observed-text candidates required before using a relation-constrained pool; otherwise fall back to the global text pool.",
    )
    parser.add_argument("--inject-text-missing-rate", type=float, default=0.0)
    parser.add_argument(
        "--text-missing-mask-strategy",
        type=str,
        choices=["random", "low_degree", "high_degree"],
        default="random",
    )
    parser.add_argument("--text-missing-mask-path", type=str, default=None)
    parser.add_argument("--text-missing-mask-seed", type=int, default=0)
    parser.add_argument("--save-text-missing-mask-path", type=str, default=None)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument(
        "--max-query-entities",
        type=int,
        default=0,
        help="Limit missing-text query entities for quick diagnostics. 0 evaluates all missing-text entities.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def read_count(path):
    with open(path, "r", encoding="utf-8") as fin:
        return int(fin.readline().strip())


def read_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 4:
            parts = parts[1:]
        h, t, r = map(int, parts[:3])
        triples.append((h, t, r))
    return triples


def build_graph_context(benchmark_path, ent_tot):
    incident_relations = [set() for _ in range(ent_tot)]
    one_hop_neighbors = [set() for _ in range(ent_tot)]
    triple_count = 0
    for split_name in ["train2id.txt"]:
        for h, t, r in read_triples(os.path.join(benchmark_path, split_name)):
            if 0 <= h < ent_tot and 0 <= t < ent_tot:
                incident_relations[h].add(r)
                incident_relations[t].add(r)
                one_hop_neighbors[h].add(t)
                one_hop_neighbors[t].add(h)
                triple_count += 1
    return incident_relations, one_hop_neighbors, triple_count


def build_relation_candidate_lookup(incident_relations, observed_text_ids):
    relation_to_candidates = {}
    for entity_id in observed_text_ids.cpu().tolist():
        for relation_id in incident_relations[entity_id]:
            relation_to_candidates.setdefault(relation_id, set()).add(entity_id)
    return {
        relation_id: torch.tensor(sorted(entity_ids), dtype=torch.long)
        for relation_id, entity_ids in relation_to_candidates.items()
    }


def limit_retrieval_pool(candidate_ids, pool_size):
    if pool_size > 0 and candidate_ids.numel() > pool_size:
        positions = torch.linspace(
            0,
            candidate_ids.numel() - 1,
            steps=pool_size,
            dtype=torch.long,
        )
        candidate_ids = candidate_ids.index_select(0, positions)
    return candidate_ids


def deterministic_random_scores(query_ids, candidate_ids):
    entity_key = query_ids.to(dtype=torch.float32).view(-1, 1) + 1.0
    candidate_key = candidate_ids.to(dtype=torch.float32).view(1, -1) + 1.0
    scores = torch.sin(entity_key * 12.9898 + candidate_key * 78.233) * 43758.5453
    return scores - torch.floor(scores)


def jaccard(left, right):
    if not left and not right:
        return None
    union_count = len(left | right)
    if union_count == 0:
        return None
    return len(left & right) / union_count


def mean_or_none(values):
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def summarize_numbers(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None, None
    return float(np.mean(clean)), float(np.min(clean)), float(np.max(clean))


def collect_neighbor_metrics(
    query_id,
    neighbors,
    incident_relations,
    one_hop_neighbors,
    relation_jaccards,
    relation_overlap_hits,
    direct_neighbor_hits,
):
    query_relations = incident_relations[query_id]
    query_neighbors = one_hop_neighbors[query_id]
    for neighbor_id in neighbors:
        score = jaccard(query_relations, incident_relations[neighbor_id])
        relation_jaccards.append(score)
        relation_overlap_hits.append(1.0 if score is not None and score > 0.0 else 0.0)
        direct_neighbor_hits.append(1.0 if neighbor_id in query_neighbors else 0.0)


def relation_constrained_pool_for_query(
    query_id,
    relation_to_candidates,
    incident_relations,
    global_candidate_ids,
    pool_size,
    min_pool_size,
):
    relation_pool = set()
    for relation_id in incident_relations[query_id]:
        candidates = relation_to_candidates.get(relation_id)
        if candidates is not None:
            relation_pool.update(candidates.cpu().tolist())
    relation_pool.discard(query_id)
    relation_pool_size_before_fallback = len(relation_pool)
    if relation_pool_size_before_fallback < min_pool_size:
        return global_candidate_ids, True, relation_pool_size_before_fallback
    candidate_ids = torch.tensor(sorted(relation_pool), dtype=torch.long)
    return limit_retrieval_pool(candidate_ids, pool_size), False, relation_pool_size_before_fallback


def evaluate_relation_constrained_source(
    entity_embeddings,
    query_ids,
    global_candidate_ids,
    observed_text_ids,
    topk,
    pool_size,
    relation_min_pool_size,
    incident_relations,
    one_hop_neighbors,
):
    entity_embeddings = F.normalize(entity_embeddings.float(), p=2, dim=-1, eps=1e-12)
    relation_to_candidates = build_relation_candidate_lookup(incident_relations, observed_text_ids)
    topk_similarity_means = []
    topk_similarity_maxes = []
    relation_jaccards = []
    relation_overlap_hits = []
    direct_neighbor_hits = []
    candidate_counts = []
    relation_pool_sizes_before_fallback = []
    fallback_count = 0
    retrieved_entity_count = 0

    for query_id in tqdm(query_ids.cpu().tolist(), desc="retrieval-quality relation_constrained_knn", leave=False):
        candidate_ids, used_fallback, relation_pool_size = relation_constrained_pool_for_query(
            query_id,
            relation_to_candidates,
            incident_relations,
            global_candidate_ids,
            pool_size,
            relation_min_pool_size,
        )
        if candidate_ids.numel() == 0:
            continue
        if used_fallback:
            fallback_count += 1
        relation_pool_sizes_before_fallback.append(relation_pool_size)
        candidate_counts.append(int(candidate_ids.numel()))
        effective_topk = min(int(topk), int(candidate_ids.numel()))
        query_embedding = entity_embeddings.index_select(0, torch.tensor([query_id], dtype=torch.long))
        candidate_embeddings = entity_embeddings.index_select(0, candidate_ids)
        similarity = torch.matmul(query_embedding, candidate_embeddings.transpose(0, 1))
        topk_scores, topk_positions = torch.topk(similarity, k=effective_topk, dim=-1)
        retrieved_ids = candidate_ids.index_select(0, topk_positions.reshape(-1)).cpu().tolist()
        topk_similarity_means.append(float(topk_scores.mean().item()))
        topk_similarity_maxes.append(float(topk_scores.max().item()))
        collect_neighbor_metrics(
            query_id,
            retrieved_ids,
            incident_relations,
            one_hop_neighbors,
            relation_jaccards,
            relation_overlap_hits,
            direct_neighbor_hits,
        )
        retrieved_entity_count += len(retrieved_ids)

    avg_candidate_count, min_candidate_count, max_candidate_count = summarize_numbers(candidate_counts)
    avg_relation_pool_size, min_relation_pool_size, max_relation_pool_size = summarize_numbers(
        relation_pool_sizes_before_fallback
    )
    return {
        "retrieval_source": "relation_constrained_knn",
        "query_entity_count": int(query_ids.numel()),
        "candidate_entity_count": int(global_candidate_ids.numel()),
        "retrieved_entity_count": int(retrieved_entity_count),
        "topk": int(topk),
        "topk_similarity_mean": mean_or_none(topk_similarity_means),
        "topk_similarity_max_mean": mean_or_none(topk_similarity_maxes),
        "incident_relation_jaccard_mean": mean_or_none(relation_jaccards),
        "incident_relation_overlap_rate": mean_or_none(relation_overlap_hits),
        "direct_train_neighbor_rate": mean_or_none(direct_neighbor_hits),
        "avg_candidate_entity_count": avg_candidate_count,
        "min_candidate_entity_count": min_candidate_count,
        "max_candidate_entity_count": max_candidate_count,
        "avg_relation_pool_size_before_fallback": avg_relation_pool_size,
        "min_relation_pool_size_before_fallback": min_relation_pool_size,
        "max_relation_pool_size_before_fallback": max_relation_pool_size,
        "fallback_entity_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(int(query_ids.numel()), 1)),
        "relation_min_pool_size": int(relation_min_pool_size),
    }


def evaluate_source(
    source,
    entity_embeddings,
    query_ids,
    candidate_ids,
    topk,
    query_batch_size,
    incident_relations,
    one_hop_neighbors,
):
    entity_embeddings = F.normalize(entity_embeddings.float(), p=2, dim=-1, eps=1e-12)
    candidate_embeddings = entity_embeddings.index_select(0, candidate_ids)
    topk_similarity_means = []
    topk_similarity_maxes = []
    relation_jaccards = []
    relation_overlap_hits = []
    direct_neighbor_hits = []
    retrieved_entity_count = 0

    for start in tqdm(range(0, query_ids.numel(), query_batch_size), desc=f"retrieval-quality {source}", leave=False):
        batch_query_ids = query_ids[start:start + query_batch_size]
        query_embeddings = entity_embeddings.index_select(0, batch_query_ids)
        similarity = torch.matmul(query_embeddings, candidate_embeddings.transpose(0, 1))
        if source == "entity_embedding_knn":
            topk_scores, topk_positions = torch.topk(similarity, k=topk, dim=-1)
        elif source == "random_text_pool":
            random_scores = deterministic_random_scores(batch_query_ids, candidate_ids)
            _unused_scores, topk_positions = torch.topk(random_scores, k=topk, dim=-1)
            topk_scores = torch.gather(similarity, dim=1, index=topk_positions)
        else:
            raise ValueError("Unsupported retrieval source: %s" % source)

        retrieved_ids = candidate_ids.index_select(0, topk_positions.reshape(-1)).view(batch_query_ids.shape[0], topk)
        topk_similarity_means.extend(topk_scores.mean(dim=-1).cpu().tolist())
        topk_similarity_maxes.extend(topk_scores.max(dim=-1).values.cpu().tolist())
        for query_id, neighbors in zip(batch_query_ids.cpu().tolist(), retrieved_ids.cpu().tolist()):
            collect_neighbor_metrics(
                query_id,
                neighbors,
                incident_relations,
                one_hop_neighbors,
                relation_jaccards,
                relation_overlap_hits,
                direct_neighbor_hits,
            )
            retrieved_entity_count += len(neighbors)

    return {
        "retrieval_source": source,
        "query_entity_count": int(query_ids.numel()),
        "candidate_entity_count": int(candidate_ids.numel()),
        "retrieved_entity_count": int(retrieved_entity_count),
        "topk": int(topk),
        "topk_similarity_mean": mean_or_none(topk_similarity_means),
        "topk_similarity_max_mean": mean_or_none(topk_similarity_maxes),
        "incident_relation_jaccard_mean": mean_or_none(relation_jaccards),
        "incident_relation_overlap_rate": mean_or_none(relation_overlap_hits),
        "direct_train_neighbor_rate": mean_or_none(direct_neighbor_hits),
        "avg_candidate_entity_count": float(candidate_ids.numel()),
        "min_candidate_entity_count": int(candidate_ids.numel()),
        "max_candidate_entity_count": int(candidate_ids.numel()),
        "fallback_entity_count": 0,
        "fallback_rate": 0.0,
    }


def main():
    args = parse_args()
    if args.retrieval_topk <= 0:
        raise ValueError("--retrieval-topk must be > 0.")
    if args.retrieval_pool_size < 0:
        raise ValueError("--retrieval-pool-size must be >= 0.")
    if args.relation_min_pool_size <= 0:
        raise ValueError("--relation-min-pool-size must be > 0.")
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be > 0.")
    if args.max_query_entities < 0:
        raise ValueError("--max-query-entities must be >= 0.")
    if args.text_missing_mask_path is not None and not os.path.exists(args.text_missing_mask_path):
        raise FileNotFoundError(f"Text-missing mask file not found: {args.text_missing_mask_path}")

    benchmark_path = os.path.join(".", "benchmarks", args.dataset)
    textual_path = os.path.join(".", "embeddings", f"{args.dataset}-textual.pth")
    if not os.path.isdir(benchmark_path):
        raise FileNotFoundError(f"Dataset directory not found: {benchmark_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    text_emb = torch.load(textual_path, map_location="cpu")
    text_emb, injection_info, injected_text_mask = apply_text_missing_injection(
        text_emb,
        args.inject_text_missing_rate,
        seed=args.text_missing_mask_seed,
        mask_strategy=args.text_missing_mask_strategy,
        benchmark_path=benchmark_path,
        mask_path=args.text_missing_mask_path,
        save_mask_path=args.save_text_missing_mask_path,
    )
    has_text = text_emb.float().norm(dim=1).ne(0).cpu()
    ent_tot = read_count(os.path.join(benchmark_path, "entity2id.txt"))
    if int(text_emb.shape[0]) != ent_tot:
        raise ValueError("Text embedding entity count does not match entity2id.txt.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "ent_embeddings.weight" not in checkpoint:
        raise KeyError("Checkpoint does not contain ent_embeddings.weight.")
    entity_embeddings = checkpoint["ent_embeddings.weight"].detach().cpu()
    if int(entity_embeddings.shape[0]) != ent_tot:
        raise ValueError("Checkpoint entity embedding count does not match dataset entity count.")

    query_ids = torch.nonzero(~has_text, as_tuple=False).view(-1)
    if args.max_query_entities > 0 and query_ids.numel() > args.max_query_entities:
        rng = np.random.default_rng(args.text_missing_mask_seed)
        selected = rng.choice(query_ids.numel(), size=args.max_query_entities, replace=False)
        query_ids = query_ids[torch.as_tensor(np.sort(selected), dtype=torch.long)]
    observed_text_ids = torch.nonzero(has_text, as_tuple=False).view(-1)
    candidate_ids = limit_retrieval_pool(observed_text_ids, args.retrieval_pool_size)
    if candidate_ids.numel() == 0:
        raise ValueError("No observed-text candidate entities are available for retrieval.")
    topk = min(int(args.retrieval_topk), int(candidate_ids.numel()))

    incident_relations, one_hop_neighbors, train_triple_count = build_graph_context(benchmark_path, ent_tot)
    if args.retrieval_source == "both":
        sources = ["entity_embedding_knn", "random_text_pool"]
    elif args.retrieval_source == "all":
        sources = ["entity_embedding_knn", "random_text_pool", "relation_constrained_knn"]
    else:
        sources = [args.retrieval_source]

    quality_by_source = {}
    for source in sources:
        if source == "relation_constrained_knn":
            quality_by_source[source] = evaluate_relation_constrained_source(
                entity_embeddings,
                query_ids,
                candidate_ids,
                observed_text_ids,
                topk,
                args.retrieval_pool_size,
                args.relation_min_pool_size,
                incident_relations,
                one_hop_neighbors,
            )
        else:
            quality_by_source[source] = evaluate_source(
                source,
                entity_embeddings,
                query_ids,
                candidate_ids,
                topk,
                args.query_batch_size,
                incident_relations,
                one_hop_neighbors,
            )

    result_payload = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "entity_count": ent_tot,
        "train_triple_count": int(train_triple_count),
        "retrieval_topk": args.retrieval_topk,
        "retrieval_pool_size": args.retrieval_pool_size,
        "relation_min_pool_size": args.relation_min_pool_size,
        "retrieval_source": args.retrieval_source,
        "inject_text_missing_rate": args.inject_text_missing_rate,
        "text_missing_mask_strategy": args.text_missing_mask_strategy,
        "text_missing_mask_seed": args.text_missing_mask_seed,
        "text_missing_mask_path": os.path.abspath(args.text_missing_mask_path) if args.text_missing_mask_path is not None else None,
        "save_text_missing_mask_path": os.path.abspath(args.save_text_missing_mask_path) if args.save_text_missing_mask_path is not None else None,
        "injection_info": injection_info,
        "injected_text_count": int(injected_text_mask.sum().item()) if injected_text_mask is not None else 0,
        "missing_text_count": int((~has_text).sum().item()),
        "observed_text_count": int(has_text.sum().item()),
        "max_query_entities": args.max_query_entities,
        "quality_by_source": quality_by_source,
    }
    if args.output_json is not None:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fout:
            json.dump(result_payload, fout, ensure_ascii=False, indent=2, sort_keys=True)
    print("RESULT_JSON: " + json.dumps(result_payload, sort_keys=True))


if __name__ == "__main__":
    main()
