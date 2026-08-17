import hashlib
import json
import os

import numpy as np
import torch


MASK_SCOPE = "entity_level_shared_across_train_valid_test"
MASK_STRATEGIES = ("random", "low_degree", "high_degree")


def _read_triples(path):
    triples = []
    if not os.path.exists(path):
        return triples
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


def compute_entity_degrees(benchmark_path, num_entities, split_names=("train2id.txt",)):
    degrees = torch.zeros(num_entities, dtype=torch.long)
    observed_triple_count = 0
    for split_name in split_names:
        path = os.path.join(benchmark_path, split_name)
        for h, t, _r in _read_triples(path):
            if 0 <= h < num_entities:
                degrees[h] += 1
            if 0 <= t < num_entities:
                degrees[t] += 1
            observed_triple_count += 1
    return degrees, {
        "degree_source_splits": list(split_names),
        "degree_observed_triple_count": int(observed_triple_count),
    }


def mask_checksum(mask):
    mask_np = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    packed = np.packbits(mask_np)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _mask_from_indices(indices, num_entities):
    mask = torch.zeros(num_entities, dtype=torch.bool)
    if len(indices) > 0:
        ids = torch.as_tensor(indices, dtype=torch.long).view(-1)
        if bool((ids < 0).any().item()) or bool((ids >= num_entities).any().item()):
            raise ValueError("Missing-text mask contains entity ids outside the dataset range.")
        mask[ids] = True
    return mask


def load_missing_text_mask(path, num_entities):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing-text mask file not found: {path}")

    metadata = {}
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fin:
            payload = json.load(fin)
    else:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        metadata = dict(payload.get("metadata") or {})
        if "mask" in payload:
            mask = torch.as_tensor(payload["mask"], dtype=torch.bool).view(-1).cpu()
        elif "indices" in payload:
            mask = _mask_from_indices(payload["indices"], num_entities)
        elif "masked_entity_ids" in payload:
            mask = _mask_from_indices(payload["masked_entity_ids"], num_entities)
        else:
            raise KeyError("Missing-text mask file must contain 'mask', 'indices', or 'masked_entity_ids'.")
    elif torch.is_tensor(payload):
        if payload.dtype == torch.bool and payload.numel() == num_entities:
            mask = payload.detach().cpu().bool().view(-1)
        else:
            mask = _mask_from_indices(payload.detach().cpu().long().view(-1).tolist(), num_entities)
    elif isinstance(payload, (list, tuple)):
        mask = _mask_from_indices(payload, num_entities)
    else:
        raise TypeError("Unsupported missing-text mask payload type: %s" % type(payload).__name__)

    if mask.numel() != num_entities:
        raise ValueError(
            "Missing-text mask length mismatch: expected %d entities, got %d."
            % (num_entities, mask.numel())
        )
    return mask, metadata


def save_missing_text_mask(path, mask, metadata):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    mask = mask.detach().cpu().bool().view(-1)
    masked_entity_ids = torch.nonzero(mask, as_tuple=False).view(-1).cpu().tolist()
    payload = {
        "mask": mask,
        "masked_entity_ids": masked_entity_ids,
        "metadata": dict(metadata),
    }
    if path.endswith(".json"):
        json_payload = {
            "masked_entity_ids": masked_entity_ids,
            "metadata": dict(metadata),
        }
        with open(path, "w", encoding="utf-8") as fout:
            json.dump(json_payload, fout, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        torch.save(payload, path)


def _degree_summary(degrees, entity_ids):
    if entity_ids.numel() == 0:
        return {
            "count": 0,
            "degree_mean": None,
            "degree_min": None,
            "degree_max": None,
        }
    values = degrees.index_select(0, entity_ids.cpu()).float()
    return {
        "count": int(entity_ids.numel()),
        "degree_mean": float(values.mean().item()),
        "degree_min": int(values.min().item()),
        "degree_max": int(values.max().item()),
    }


def _select_mask_indices(available_indices, mask_count, strategy, seed, benchmark_path, num_entities):
    if strategy not in MASK_STRATEGIES:
        raise ValueError("--text-missing-mask-strategy must be one of: %s" % ", ".join(MASK_STRATEGIES))

    rng = np.random.default_rng(seed)
    available_count = int(available_indices.numel())
    if mask_count <= 0:
        return available_indices[:0], None

    if strategy == "random":
        selected = rng.choice(available_count, size=mask_count, replace=False)
        return available_indices[torch.as_tensor(selected, dtype=torch.long)], None

    if benchmark_path is None:
        raise ValueError("--text-missing-mask-strategy=%s requires benchmark_path." % strategy)

    degrees, degree_info = compute_entity_degrees(benchmark_path, num_entities)
    available_degrees = degrees.index_select(0, available_indices.cpu()).numpy()
    tie_break = rng.random(available_count)
    primary = available_degrees if strategy == "low_degree" else -available_degrees
    order = np.lexsort((tie_break, primary))
    selected = order[:mask_count]
    selected_indices = available_indices[torch.as_tensor(selected, dtype=torch.long)]
    degree_info.update(
        {
            "degree_strategy": strategy,
            "available_text_degree_summary": _degree_summary(degrees, available_indices.cpu()),
            "masked_text_degree_summary": _degree_summary(degrees, selected_indices.cpu()),
        }
    )
    return selected_indices, degree_info


def build_text_missing_mask(
    text_emb,
    missing_rate,
    seed=0,
    mask_strategy="random",
    benchmark_path=None,
    mask_path=None,
):
    if missing_rate < 0.0 or missing_rate > 1.0:
        raise ValueError("Text missing rate must be in [0, 1].")

    num_entities = int(text_emb.shape[0])
    original_has_text = text_emb.float().norm(dim=1).ne(0).cpu()
    available_indices = torch.nonzero(original_has_text, as_tuple=False).view(-1)
    available_count = int(available_indices.numel())

    source_metadata = {}
    if mask_path is not None:
        mask, source_metadata = load_missing_text_mask(mask_path, num_entities)
        unavailable_masked = mask & (~original_has_text)
        if bool(unavailable_masked.any().item()):
            raise ValueError(
                "Loaded missing-text mask selects %d entities without original text."
                % int(unavailable_masked.sum().item())
            )
        mask_source = "loaded_file"
        degree_info = None
    else:
        mask_count = int(round(available_count * missing_rate))
        selected_indices, degree_info = _select_mask_indices(
            available_indices,
            mask_count,
            mask_strategy,
            seed,
            benchmark_path,
            num_entities,
        )
        mask = torch.zeros(num_entities, dtype=torch.bool)
        if selected_indices.numel() > 0:
            mask[selected_indices.cpu()] = True
        mask_source = "generated"

    masked_count = int(mask.sum().item())
    info = {
        "mask_scope": MASK_SCOPE,
        "missingness_unit": "entity",
        "mask_source": mask_source,
        "mask_strategy": mask_strategy,
        "mask_path": os.path.abspath(mask_path) if mask_path is not None else None,
        "seed": int(seed),
        "inject_rate_requested": float(missing_rate),
        "inject_rate_applied": masked_count / available_count if available_count > 0 else 0.0,
        "available_text_before": available_count,
        "additional_masked_count": masked_count,
        "mask_checksum_sha256": mask_checksum(mask),
        "masked_entity_ids_preview": torch.nonzero(mask, as_tuple=False).view(-1)[:20].cpu().tolist(),
        "loaded_mask_metadata": source_metadata,
    }
    if degree_info is not None:
        info.update(degree_info)
    return mask, info


def apply_text_missing_injection(
    text_emb,
    inject_rate,
    seed=0,
    mask_strategy="random",
    benchmark_path=None,
    mask_path=None,
    save_mask_path=None,
):
    if inject_rate <= 0.0 and mask_path is None:
        return text_emb, None, None

    injected_text_emb = text_emb.clone()
    injected_text_mask, info = build_text_missing_mask(
        text_emb,
        inject_rate,
        seed=seed,
        mask_strategy=mask_strategy,
        benchmark_path=benchmark_path,
        mask_path=mask_path,
    )
    injected_text_emb[injected_text_mask] = 0
    if save_mask_path is not None:
        save_info = dict(info)
        save_info["save_mask_path"] = os.path.abspath(save_mask_path)
        save_missing_text_mask(save_mask_path, injected_text_mask, save_info)
        info["save_mask_path"] = os.path.abspath(save_mask_path)
    else:
        info["save_mask_path"] = None
    return injected_text_emb, info, injected_text_mask


def apply_simulated_native_text_missing(text_emb, missing_rate, seed=0):
    simulated_text_emb, info, simulated_mask = apply_text_missing_injection(
        text_emb,
        missing_rate,
        seed=seed,
        mask_strategy="random",
    )
    if info is None:
        return simulated_text_emb, None, None
    native_info = {
        "mask_scope": info["mask_scope"],
        "missingness_unit": info["missingness_unit"],
        "mask_source": info["mask_source"],
        "mask_strategy": info["mask_strategy"],
        "mask_checksum_sha256": info["mask_checksum_sha256"],
        "simulate_rate_requested": info["inject_rate_requested"],
        "simulate_rate_applied": info["inject_rate_applied"],
        "available_text_before": info["available_text_before"],
        "simulated_native_missing_count": info["additional_masked_count"],
        "masked_entity_ids_preview": info["masked_entity_ids_preview"],
    }
    return simulated_text_emb, native_info, simulated_mask
