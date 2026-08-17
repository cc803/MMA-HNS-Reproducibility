# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class Trainer_dhns(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 train_mode='adp',
                 beta=0.5,
                 generator=None,
                 lrg=None,
                 mu=None,
                 g_epoch=100,
                 debug_masking=False,
                 debug_mask_batches=0,
                 debug_reliability=False,
                 debug_reliability_batches=0,
                 use_learned_reliability_conditioning=False,
                 use_text_loss_gating=False,
                 use_text_sampling_gating=False,
                 missing_sample_weight=1.0,
                 use_missing_text_aux_loss=False,
                 missing_text_aux_weight=0.1,
                 record_missing_token_diagnostics=False,
                 diagnostic_epoch_interval=1,
                 diagnostic_entity_sample_size=2048,
                 diagnostic_batch_size=4096,
                 diagnostic_seed=0):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

        self.train_mode = train_mode
        self.beta = beta

        # learning rate of the generator
        assert lrg is not None
        self.alpha_g = lrg

        # the generator part
        assert generator is not None
        assert mu is not None
        self.optimizer_g = None
        self.generator = generator
        self.batch_size = self.model.batch_size
        self.generator.cuda()
        self.mu = mu
        self.g_epoch = g_epoch
        self.debug_masking = debug_masking
        self.debug_mask_batches = debug_mask_batches
        self.debug_mask_step = 0
        self.debug_reliability = debug_reliability
        self.debug_reliability_batches = debug_reliability_batches
        self.debug_reliability_step = 0
        self.use_learned_reliability_conditioning = use_learned_reliability_conditioning
        self.use_text_loss_gating = use_text_loss_gating
        self.use_text_sampling_gating = use_text_sampling_gating
        self.debug_text_sampling_step = 0
        self.missing_sample_weight = missing_sample_weight
        self.missing_sample_weight_debug_step = 0
        self.use_missing_text_aux_loss = use_missing_text_aux_loss
        self.missing_text_aux_weight = missing_text_aux_weight
        self.current_batch_missing_text_count = 0
        self.aux_loss_value = 0.0
        self.record_missing_token_diagnostics = record_missing_token_diagnostics
        self.diagnostic_epoch_interval = max(1, int(diagnostic_epoch_interval))
        self.diagnostic_entity_sample_size = int(diagnostic_entity_sample_size)
        self.diagnostic_batch_size = max(1, int(diagnostic_batch_size))
        self.diagnostic_seed = int(diagnostic_seed)
        self.missing_token_diagnostics = []
        self._diagnostic_entity_ids = None
        self._epoch_missing_token_grad_records = []

    def _text_branch_enabled(self):
        model_core = getattr(self.model, "model", None)
        if model_core is None:
            return True
        return getattr(model_core, "uses_text_branch", lambda: True)()

    def _get_text_pair_mask(self, batch_h, batch_t):
        if not self._text_branch_enabled():
            return None
        has_text = getattr(self.model.model, "has_text", None)
        if has_text is None:
            return None
        head_ids = batch_h.detach().view(-1).cpu()
        tail_ids = batch_t.detach().view(-1).cpu()
        pair_mask = has_text.index_select(0, head_ids) & has_text.index_select(0, tail_ids)
        return pair_mask.to(device=batch_h.device, dtype=torch.bool)

    def _get_missing_text_sample_mask(self, batch_h, batch_t):
        if not self._text_branch_enabled():
            return None
        has_text = getattr(self.model.model, "has_text", None)
        if has_text is None:
            return None
        head_ids = batch_h.detach().view(-1).cpu()
        tail_ids = batch_t.detach().view(-1).cpu()
        head_has_text = has_text.index_select(0, head_ids).to(device=batch_h.device, dtype=torch.bool)
        tail_has_text = has_text.index_select(0, tail_ids).to(device=batch_t.device, dtype=torch.bool)
        return ~(head_has_text & tail_has_text)

    def _compute_missing_text_aux_loss(self, batch_h, batch_t, p_score=None, n_score=None):
        self.current_batch_missing_text_count = 0
        self.aux_loss_value = 0.0
        if not self.use_missing_text_aux_loss:
            return None
        missing_mask = self._get_missing_text_sample_mask(batch_h, batch_t)
        if missing_mask is None or p_score is None or n_score is None:
            if p_score is not None:
                return p_score.sum() * 0.0
            return None
        self.current_batch_missing_text_count = int(missing_mask.sum().item())
        if self.current_batch_missing_text_count == 0:
            aux_loss = p_score.sum() * 0.0
        else:
            aux_weight = missing_mask.to(device=p_score.device, dtype=p_score.dtype)
            aux_loss = self.model.loss(p_score, n_score, sample_weight=aux_weight)
        self.aux_loss_value = float(aux_loss.detach().item())
        return aux_loss

    def get_missing_text_aux_state(self):
        return {
            "enabled": bool(self.use_missing_text_aux_loss),
            "missing_text_aux_weight": self.missing_text_aux_weight,
            "current_batch_missing_text_count": int(self.current_batch_missing_text_count),
            "aux_loss_value": float(self.aux_loss_value),
        }

    def get_missing_token_diagnostics_state(self):
        return {
            "enabled": bool(self.record_missing_token_diagnostics),
            "diagnostic_epoch_interval": int(self.diagnostic_epoch_interval),
            "diagnostic_entity_sample_size": int(self.diagnostic_entity_sample_size),
            "diagnostic_batch_size": int(self.diagnostic_batch_size),
            "diagnostic_seed": int(self.diagnostic_seed),
            "history": list(self.missing_token_diagnostics),
            "last": self.missing_token_diagnostics[-1] if self.missing_token_diagnostics else None,
        }

    def _summarize_values(self, values):
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

    def _summarize_tensor_values(self, tensor):
        if tensor is None:
            return self._summarize_values([])
        flat = tensor.detach().view(-1).float().cpu()
        return self._summarize_values(flat.tolist())

    def _select_diagnostic_entity_ids(self):
        if self._diagnostic_entity_ids is not None:
            return self._diagnostic_entity_ids
        model_core = getattr(self.model, "model", None)
        has_text = getattr(model_core, "has_text", None)
        if has_text is None:
            self._diagnostic_entity_ids = torch.empty(0, dtype=torch.long)
            return self._diagnostic_entity_ids

        has_text = has_text.detach().cpu().bool().view(-1)
        missing_ids = torch.nonzero(~has_text, as_tuple=False).view(-1)
        observed_ids = torch.nonzero(has_text, as_tuple=False).view(-1)
        sample_size = self.diagnostic_entity_sample_size

        def _sample(ids, seed_offset):
            if sample_size <= 0 or ids.numel() <= sample_size:
                return ids
            generator = torch.Generator()
            generator.manual_seed(self.diagnostic_seed + seed_offset)
            order = torch.randperm(ids.numel(), generator=generator)[:sample_size]
            return ids.index_select(0, order)

        sampled_missing = _sample(missing_ids, 1009)
        sampled_observed = _sample(observed_ids, 2003)
        self._diagnostic_entity_ids = torch.unique(
            torch.cat([sampled_missing, sampled_observed], dim=0),
            sorted=True,
        )
        return self._diagnostic_entity_ids

    def _record_missing_token_gradients(self, batch_h, batch_t):
        if not self.record_missing_token_diagnostics:
            return
        model_core = getattr(self.model, "model", None)
        if model_core is None:
            return

        grad_norms = {}
        for param_name in [
            "missing_text_token",
            "missing_text_token_head",
            "missing_text_token_tail",
            "missing_text_token_scale",
        ]:
            param = getattr(model_core, param_name, None)
            if param is None or param.grad is None:
                grad_norms[param_name] = None
            else:
                grad_norms[param_name] = float(param.grad.detach().norm().item())

        missing_mask = self._get_missing_text_sample_mask(batch_h, batch_t)
        missing_sample_count = int(missing_mask.sum().item()) if missing_mask is not None else 0
        self._epoch_missing_token_grad_records.append({
            "missing_related_sample_count": missing_sample_count,
            "grad_norms": grad_norms,
        })

    def _summarize_epoch_gradients(self):
        records = self._epoch_missing_token_grad_records
        summary = {
            "batch_count": int(len(records)),
            "missing_related_sample_count": int(sum(record["missing_related_sample_count"] for record in records)),
            "tokens": {},
        }
        for param_name in [
            "missing_text_token",
            "missing_text_token_head",
            "missing_text_token_tail",
            "missing_text_token_scale",
        ]:
            values = [record["grad_norms"].get(param_name) for record in records]
            token_summary = self._summarize_values(values)
            token_summary["nonzero_count"] = int(
                sum(1 for value in values if value is not None and np.isfinite(float(value)) and float(value) > 0.0)
            )
            summary["tokens"][param_name] = token_summary
        return summary

    def _collect_attention_diagnostics(self):
        model_core = getattr(self.model, "model", None)
        if model_core is None or not hasattr(model_core, "export_fusion_probe"):
            return {
                "enabled": False,
                "reason": "fusion_probe_unavailable",
            }
        if not getattr(model_core, "uses_text_branch", lambda: True)():
            return {
                "enabled": False,
                "reason": "text_branch_disabled",
            }

        entity_ids = self._select_diagnostic_entity_ids()
        if entity_ids.numel() == 0:
            return {
                "enabled": False,
                "reason": "no_entities_for_probe",
            }

        probe = model_core.export_fusion_probe(entity_ids, batch_size=self.diagnostic_batch_size)
        modalities = probe.get("modalities", [])
        if "text" not in modalities:
            return {
                "enabled": False,
                "reason": "text_attention_unavailable",
            }

        text_index = modalities.index("text")
        beta_text = probe["attention"][:, text_index].float()
        has_text = probe["has_text"].bool()
        missing_mask = ~has_text
        observed_mask = has_text
        return {
            "enabled": True,
            "sample_entity_count": int(entity_ids.numel()),
            "missing_entity_count": int(missing_mask.sum().item()),
            "observed_entity_count": int(observed_mask.sum().item()),
            "text_beta_all": self._summarize_tensor_values(beta_text),
            "text_beta_missing": self._summarize_tensor_values(beta_text[missing_mask]),
            "text_beta_observed": self._summarize_tensor_values(beta_text[observed_mask]),
        }

    def _should_record_epoch_diagnostics(self, epoch):
        if not self.record_missing_token_diagnostics:
            return False
        return (
            epoch == 0 or
            (epoch + 1) == self.train_times or
            (epoch + 1) % self.diagnostic_epoch_interval == 0
        )

    def _collect_epoch_diagnostics(self, epoch, kgc_loss, diffusion_loss):
        if not self._should_record_epoch_diagnostics(epoch):
            self._epoch_missing_token_grad_records = []
            return
        record = {
            "epoch": int(epoch),
            "epoch_number": int(epoch + 1),
            "kgc_loss": float(kgc_loss),
            "diffusion_loss": float(diffusion_loss),
            "gradients": self._summarize_epoch_gradients(),
            "attention": self._collect_attention_diagnostics(),
        }
        self.missing_token_diagnostics.append(record)
        attention = record["attention"]
        grad = record["gradients"]["tokens"].get("missing_text_token", {})
        beta_missing = None
        if attention.get("enabled"):
            beta_missing = attention["text_beta_missing"].get("mean")
        print(
            "Missing-token diagnostics | epoch=%d | beta_text_missing=%s | missing_text_token_grad_mean=%s"
            % (
                epoch,
                "NA" if beta_missing is None else "%.6f" % beta_missing,
                "NA" if grad.get("mean") is None else "%.6f" % grad["mean"],
            )
        )
        self._epoch_missing_token_grad_records = []

    def _get_missing_sample_weight(self, batch_h, batch_t):
        if self.missing_sample_weight <= 1.0:
            return None
        has_text = getattr(self.model.model, "has_text", None)
        if has_text is None:
            return None
        head_ids = batch_h.detach().view(-1).cpu()
        tail_ids = batch_t.detach().view(-1).cpu()
        head_has_text = has_text.index_select(0, head_ids).to(device=batch_h.device, dtype=torch.bool)
        tail_has_text = has_text.index_select(0, tail_ids).to(device=batch_t.device, dtype=torch.bool)
        missing_related = ~(head_has_text & tail_has_text)
        sample_weight = torch.ones(batch_h.shape[0], device=batch_h.device, dtype=torch.float32)
        sample_weight[missing_related] = self.missing_sample_weight
        if self.missing_sample_weight_debug_step < 3:
            print(
                "Missing-sample weighting debug | weight=%.4f | batch_size=%d | missing_related=%d | fully_observed=%d"
                % (
                    self.missing_sample_weight,
                    int(batch_h.shape[0]),
                    int(missing_related.sum().item()),
                    int((~missing_related).sum().item()),
                )
            )
            self.missing_sample_weight_debug_step += 1
        return sample_weight

    def _apply_text_sampling_gating(self, neg_text_list, base_text_embs, text_pair_mask):
        if text_pair_mask is None:
            return neg_text_list
        if bool(text_pair_mask.all().item()):
            return neg_text_list

        gated_negatives = []
        invalid_mask = ~text_pair_mask
        for neg_text in neg_text_list:
            gated_text = neg_text.clone()
            gated_text[invalid_mask] = base_text_embs[invalid_mask]
            gated_negatives.append(gated_text)
        return gated_negatives

    def _get_missing_aware_conditioning(self, batch_h, batch_t, batch_r=None):
        if not (
            getattr(self.generator, "use_missing_aware_conditioning", False) or
            getattr(self.generator, "use_missing_aware_film_conditioning", False)
        ):
            return None

        model_core = getattr(self.model, "model", None)
        if self.use_learned_reliability_conditioning:
            if model_core is None or not hasattr(model_core, "get_learned_reliability_conditioning"):
                return None
            if batch_r is None:
                return None
            return model_core.get_learned_reliability_conditioning(batch_h, batch_t, batch_r)

        has_text = getattr(model_core, "has_text", None)
        has_image = getattr(model_core, "has_image", None)
        if has_text is None and has_image is None:
            return None

        head_ids = batch_h.detach().view(-1).cpu()
        tail_ids = batch_t.detach().view(-1).cpu()
        device = batch_h.device

        def _lookup(mask, ids):
            if mask is None:
                return torch.ones(ids.shape[0], 1, device=device, dtype=torch.float32)
            return mask.index_select(0, ids).to(device=device, dtype=torch.float32).unsqueeze(-1)

        head_image = _lookup(has_image, head_ids)
        tail_image = _lookup(has_image, tail_ids)
        if not self._text_branch_enabled():
            return torch.cat([head_image, tail_image], dim=-1)
        head_text = _lookup(has_text, head_ids)
        tail_text = _lookup(has_text, tail_ids)
        return torch.cat([head_image, head_text, tail_image, tail_text], dim=-1)

    def _mean_or_nan(self, tensor, mask=None):
        if mask is not None:
            if not bool(mask.any().item()):
                return float("nan")
            tensor = tensor[mask]
        return float(tensor.float().mean().item())

    def _debug_modality_reliability(self, batch_h, batch_t, batch_r):
        if not self.debug_reliability or self.debug_reliability_step >= self.debug_reliability_batches:
            return

        model_core = getattr(self.model, "model", None)
        if model_core is None or not hasattr(model_core, "get_modality_reliability"):
            return

        head_stats = model_core.get_modality_reliability(batch_h, batch_r, condition_mode="tail")
        tail_stats = model_core.get_modality_reliability(batch_t, batch_r, condition_mode="head")

        def _format_side(name, stats):
            text_rel = stats["text_reliability"].view(-1)
            image_rel = stats["image_reliability"].view(-1)
            text_avail = stats["text_available"].view(-1) > 0.5
            image_avail = stats["image_available"].view(-1) > 0.5
            text_missing = ~text_avail
            image_missing = ~image_avail
            return (
                f"{name}_text_rel_mean={self._mean_or_nan(text_rel):.6f} | "
                f"{name}_text_rel_avail_mean={self._mean_or_nan(text_rel, text_avail):.6f} | "
                f"{name}_text_rel_missing_mean={self._mean_or_nan(text_rel, text_missing):.6f} | "
                f"{name}_text_avail_ratio={self._mean_or_nan(stats['text_available']):.6f} | "
                f"{name}_image_rel_mean={self._mean_or_nan(image_rel):.6f} | "
                f"{name}_image_rel_avail_mean={self._mean_or_nan(image_rel, image_avail):.6f} | "
                f"{name}_image_rel_missing_mean={self._mean_or_nan(image_rel, image_missing):.6f} | "
                f"{name}_image_avail_ratio={self._mean_or_nan(stats['image_available']):.6f}"
            )

        print(f"Relation-aware reliability debug | batch {self.debug_reliability_step + 1} | " + _format_side("head", head_stats))
        print(f"Relation-aware reliability debug | batch {self.debug_reliability_step + 1} | " + _format_side("tail", tail_stats))
        self.debug_reliability_step += 1

    def _debug_text_sampling_gating(
        self,
        text_pair_mask,
        original_neg_ht,
        original_neg_tt,
        gated_neg_ht,
        gated_neg_tt,
    ):
        if self.debug_text_sampling_step >= 3:
            return

        if text_pair_mask is None:
            invalid_pairs = 0
            invalid_mask = None
        else:
            invalid_mask = ~text_pair_mask
            invalid_pairs = int(invalid_mask.sum().item())

        if invalid_mask is None or invalid_pairs == 0:
            replaced_ht_count = 0
            replaced_tt_count = 0
        else:
            replaced_ht_count = invalid_pairs * len(gated_neg_ht)
            replaced_tt_count = invalid_pairs * len(gated_neg_tt)

        ht_diff = torch.tensor(0.0, device=original_neg_ht[0].device)
        tt_diff = torch.tensor(0.0, device=original_neg_tt[0].device)
        for before, after in zip(original_neg_ht, gated_neg_ht):
            ht_diff += (after - before).abs().sum()
        for before, after in zip(original_neg_tt, gated_neg_tt):
            tt_diff += (after - before).abs().sum()

        changed = bool((ht_diff > 0).item() or (tt_diff > 0).item())
        print(
            "Text sampling gating debug | batch %d | invalid_text_pairs: %d | replaced_neg_ht: %d | replaced_neg_tt: %d | ht_l1_diff: %.6f | tt_l1_diff: %.6f | gating changed textual negatives: %s"
            % (
                self.debug_text_sampling_step + 1,
                invalid_pairs,
                replaced_ht_count,
                replaced_tt_count,
                float(ht_diff.item()),
                float(tt_diff.item()),
                "True" if changed else "False",
            )
        )
        self.debug_text_sampling_step += 1

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        batch_h_gen = self.to_var(data['batch_h'][0: self.batch_size], self.use_gpu)
        batch_t_gen = self.to_var(data['batch_t'][0: self.batch_size], self.use_gpu)
        sample_weight = self._get_missing_sample_weight(batch_h_gen, batch_t_gen)
        loss, p_score = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode'],
            'sample_weight': sample_weight,
            'capture_scores_for_aux': self.use_missing_text_aux_loss,
        })
        auxiliary_loss = getattr(self.model.model, "consume_auxiliary_loss", lambda: None)()
        if auxiliary_loss is not None:
            loss += auxiliary_loss
        aux_p_score, aux_n_score = getattr(self.model, "consume_last_scores", lambda: (None, None))()
        missing_text_aux_loss = self._compute_missing_text_aux_loss(
            batch_h_gen,
            batch_t_gen,
            p_score=aux_p_score,
            n_score=aux_n_score,
        )
        if missing_text_aux_loss is not None:
            loss += self.missing_text_aux_weight * missing_text_aux_loss

        # training DHNE generator module
        batch_r = self.to_var(data['batch_r'][0: self.batch_size], self.use_gpu)
        batch_hs = self.model.model.get_batch_ent_embs(batch_h_gen)
        batch_ts = self.model.model.get_batch_ent_embs(batch_t_gen)
        batch_r = self.model.model.get_batch_rel_embs(batch_r)
        #print("batch_r:\t", batch_r.size())
        batch_hv = self.model.model.get_batch_img_embs(batch_h_gen)
        batch_tv = self.model.model.get_batch_img_embs(batch_t_gen)
        batch_ht = self.model.model.get_batch_text_embs(batch_h_gen)
        batch_tt = self.model.model.get_batch_text_embs(batch_t_gen)
        text_branch_enabled = batch_ht is not None and batch_tt is not None
        self._debug_modality_reliability(batch_h_gen, batch_t_gen, batch_r)
        batch_missing_condition = self._get_missing_aware_conditioning(batch_h_gen, batch_t_gen, batch_r=batch_r)
        def train_diffusion():
            for epoch in range(self.g_epoch):
                self.optimizer_g.zero_grad()
                
                diff_loss = self.generator(batch_hs, batch_r, batch_ts, availability=batch_missing_condition) # structurl diffusion loss
                diff_loss += self.generator(batch_hv, batch_r, batch_tv, availability=batch_missing_condition) # visual diffusion loss
                if text_branch_enabled:
                    if self.use_text_loss_gating:
                        text_pair_mask = self._get_text_pair_mask(batch_h_gen, batch_t_gen)
                        if text_pair_mask is None:
                            diff_loss += self.generator(batch_ht, batch_r, batch_tt, availability=batch_missing_condition) # textual diffusion loss
                        elif bool(text_pair_mask.any().item()):
                            diff_loss += self.generator(
                                batch_ht[text_pair_mask],
                                batch_r[text_pair_mask],
                                batch_tt[text_pair_mask],
                                availability=batch_missing_condition[text_pair_mask]
                            ) # textual diffusion loss on valid text pairs only
                    else:
                        diff_loss += self.generator(batch_ht, batch_r, batch_tt, availability=batch_missing_condition) # textual diffusion loss
                diff_loss.backward(retain_graph=True)
                self.optimizer_g.step()
                return diff_loss
        diff_loss = train_diffusion()

        # generate multimodal semantics-guided negative samples
        batch_neg_h, batch_neg_t = self.generator.sample(batch_hs, batch_r, batch_ts, availability=batch_missing_condition)
        batch_neg_hv, batch_neg_tv = self.generator.sample(batch_hv, batch_r, batch_tv, availability=batch_missing_condition)
        batch_neg_ht = None
        batch_neg_tt = None
        if text_branch_enabled:
            batch_neg_ht, batch_neg_tt = self.generator.sample(batch_ht, batch_r, batch_tt, availability=batch_missing_condition)
        if text_branch_enabled and self.use_text_sampling_gating:
            text_pair_mask = self._get_text_pair_mask(batch_h_gen, batch_t_gen)
            original_neg_ht = [neg.clone() for neg in batch_neg_ht]
            original_neg_tt = [neg.clone() for neg in batch_neg_tt]
            batch_neg_ht = self._apply_text_sampling_gating(batch_neg_ht, batch_ht, text_pair_mask)
            batch_neg_tt = self._apply_text_sampling_gating(batch_neg_tt, batch_tt, text_pair_mask)
            self._debug_text_sampling_gating(
                text_pair_mask=text_pair_mask,
                original_neg_ht=original_neg_ht,
                original_neg_tt=original_neg_tt,
                gated_neg_ht=batch_neg_ht,
                gated_neg_tt=batch_neg_tt,
            )

        # multi-level hard negative sample-based learning
        w = [0, 0.5, 1.0, 1.0, 0.5]
        w_m = [0.1, 0.3, 0.5, 0.7, 0.9]
        neg_list = []

        for i in range(len(w)):
            scores = self.model.model.mm_negative_score(
                batch_h=batch_h_gen,
                batch_r=batch_r,
                batch_t=batch_t_gen,
                mode=data['mode'],
                w_margin=w_m[i],
                neg_h=batch_neg_h[i],
                neg_t=batch_neg_t[i],
                neg_hv=batch_neg_hv[i],
                neg_tv=batch_neg_tv[i],
                neg_ht=batch_neg_ht[i] if text_branch_enabled else None,
                neg_tt=batch_neg_tt[i] if text_branch_enabled else None
            )

            if text_branch_enabled and self.use_text_sampling_gating and i == 0 and self.debug_text_sampling_step <= 3:
                current_ht = batch_neg_ht[i]
                current_tt = batch_neg_tt[i]
                print(
                    "Text sampling gating flow | level %d into mm_negative_score | neg_ht_norm: %.6f | neg_tt_norm: %.6f"
                    % (
                        i,
                        float(current_ht.norm().item()),
                        float(current_tt.norm().item()),
                    )
                )

            if (
                i == 0 and
                self.debug_masking and
                getattr(self.model.model, "use_missing_mask", False) and
                self.debug_mask_step < self.debug_mask_batches
            ):
                debug_info = getattr(self.model.model, "last_mask_debug", None)
                if debug_info is not None:
                    print(
                        "Mask debug | batch %d | head_missing_text: %d | tail_missing_text: %d | score_all_text_masked: %d"
                        % (
                            self.debug_mask_step + 1,
                            debug_info["head_missing_text_count"],
                            debug_info["tail_missing_text_count"],
                            debug_info["score_all_text_masked_count"],
                        )
                    )
                    self.debug_mask_step += 1

            for score in scores:
                neg_list.append(self.model.loss(p_score, score, sample_weight=sample_weight) * w[i] * self.mu)
        sam = [1 for i in w if i != 0]
        loss_neg = sum(neg_list) / (sum(sam)*3)
        loss += loss_neg

        loss.backward()
        self._record_missing_token_gradients(batch_h_gen, batch_t_gen)
        self.optimizer.step()
        return loss.item(), diff_loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            res_g = 0.0
            self._epoch_missing_token_grad_records = []
            for data in self.data_loader:
                loss, loss_g = self.train_one_step(data)
                res += loss
                res_g += loss_g
            if self.use_missing_text_aux_loss:
                training_range.set_description(
                    "Epoch %d | KGC loss: %f, DiffHEG loss %f | missing_text_aux_count %d | aux_loss %.6f"
                    % (epoch, res, res_g, self.current_batch_missing_text_count, self.aux_loss_value)
                )
            else:
                training_range.set_description("Epoch %d | KGC loss: %f, DiffHEG loss %f" % (epoch, res, res_g))
            self._collect_epoch_diagnostics(epoch, res, res_g)

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
