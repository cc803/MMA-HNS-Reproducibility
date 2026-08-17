import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import os
from .Model import Model


class AdvMixRotatE(Model):

    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None,
        has_text=None,
        original_has_text=None,
        has_image=None,
        use_missing_mask=False,
        use_soft_missing_text=False,
        use_fixed_zero_missing_text=False,
        use_side_aware_missing_text=False,
        use_soft_missing_image=False,
        use_prototype_missing_text=False,
        prototype_missing_text_cluster_ids=None,
        prototype_missing_text_num_clusters=None,
        use_structure_conditioned_missing_text=False,
        use_missing_aware_joint_scoring=False,
        use_masked_fixed_denominator_joint_scoring=False,
        use_availability_router=False,
        availability_router_mode="query_masked_softmax",
        availability_router_eps=1e-6,
        debug_joint_scoring_sanity=False,
        debug_missing_aware_joint_scoring=False,
        debug_missing_aware_joint_scoring_batches=3,
        use_missing_aware_fusion=False,
        missing_text_attention_scale=1.0,
        debug_fusion_sanity=False,
        use_learnable_missing_text_gate=False,
        use_oracle_restore_injected_text=False,
        oracle_text_emb=None,
        injected_text_mask=None,
        disable_text=False,
        pseudo_missing_prob=0.0,
        use_soft_token_text_generator_alignment=False,
        use_missingness_relation_expert=False,
        expert_hidden_dim=128,
        expert_num=2,
        use_branch_local_relation_gate=False,
        branch_gate_hidden_dim=64,
        use_entity_specific_missing_text=False,
        use_retrieval_missing_text=False,
        retrieval_topk=5,
        retrieval_pool_size=512,
        retrieval_source="entity_embedding_knn",
        retrieval_mix_weight=1.0,
        use_retrieval_confidence_calibration=False,
        retrieval_confidence_type="mean_topk_similarity",
        retrieval_confidence_min=0.1,
        retrieval_confidence_max=1.0,
        use_cross_modal_text_imputer=False,
        text_imputer_hidden_dim=256,
        text_imputer_residual_weight=0.05,
        text_imputer_rec_weight=0.01,
        text_imputer_nce_weight=0.01,
        text_imputer_temperature=0.07,
        use_confidence_gated_retrieval=False,
        retrieval_gate_type="similarity_based",
        retrieval_gate_min=0.0,
        retrieval_gate_max=1.0,
        use_relation_aware_retrieval=False,
        relation_head_text_pools=None,
        relation_tail_text_pools=None,
        min_relation_pool_size=32,
        relation_retrieval_fallback="global_text_pool",
        entity_specific_missing_text_recon_weight=0.0,
        use_missing_text_consistency=False,
        consistency_prob=0.1,
        consistency_lambda=0.05,
        use_missing_text_token_scale=False,
    ):

        super(AdvMixRotatE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)
        self.text_proj = nn.Linear(self.text_dim, self.dim_e)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)
        self.oracle_text_embeddings = (
            nn.Embedding.from_pretrained(oracle_text_emb).requires_grad_(True)
            if oracle_text_emb is not None else None
        )
        self.has_text = has_text.detach().cpu().bool() if has_text is not None else None
        self.original_has_text = original_has_text.detach().cpu().bool() if original_has_text is not None else self.has_text
        self.has_image = has_image.detach().cpu().bool() if has_image is not None else None
        self.injected_text_mask = injected_text_mask.detach().cpu().bool() if injected_text_mask is not None else None
        self.use_missing_mask = use_missing_mask
        self.use_soft_missing_text = use_soft_missing_text
        self.use_fixed_zero_missing_text = use_fixed_zero_missing_text
        self.use_side_aware_missing_text = use_side_aware_missing_text
        self.use_soft_missing_image = use_soft_missing_image
        self.use_prototype_missing_text = use_prototype_missing_text
        self.use_structure_conditioned_missing_text = use_structure_conditioned_missing_text
        self.use_missing_aware_joint_scoring = use_missing_aware_joint_scoring
        self.use_masked_fixed_denominator_joint_scoring = use_masked_fixed_denominator_joint_scoring
        self.use_availability_router = use_availability_router
        self.availability_router_mode = availability_router_mode
        self.availability_router_eps = availability_router_eps
        self.debug_joint_scoring_sanity = debug_joint_scoring_sanity
        self.debug_missing_aware_joint_scoring = debug_missing_aware_joint_scoring
        self.debug_missing_aware_joint_scoring_batches = debug_missing_aware_joint_scoring_batches
        self.use_missing_aware_fusion = use_missing_aware_fusion
        self.missing_text_attention_scale = missing_text_attention_scale
        self.debug_fusion_sanity = debug_fusion_sanity
        self.use_learnable_missing_text_gate = use_learnable_missing_text_gate
        self.use_oracle_restore_injected_text = use_oracle_restore_injected_text
        self.disable_text = disable_text
        self.pseudo_missing_prob = pseudo_missing_prob
        self.use_soft_token_text_generator_alignment = use_soft_token_text_generator_alignment
        self.use_missingness_relation_expert = use_missingness_relation_expert
        self.expert_hidden_dim = expert_hidden_dim
        self.expert_num = expert_num
        self.use_branch_local_relation_gate = use_branch_local_relation_gate
        self.branch_gate_hidden_dim = branch_gate_hidden_dim
        self.use_entity_specific_missing_text = use_entity_specific_missing_text
        self.use_retrieval_missing_text = use_retrieval_missing_text
        self.retrieval_topk = retrieval_topk
        self.retrieval_pool_size = retrieval_pool_size
        self.retrieval_source = retrieval_source
        self.retrieval_mix_weight = retrieval_mix_weight
        self.use_retrieval_confidence_calibration = use_retrieval_confidence_calibration
        self.retrieval_confidence_type = retrieval_confidence_type
        self.retrieval_confidence_min = retrieval_confidence_min
        self.retrieval_confidence_max = retrieval_confidence_max
        self.use_cross_modal_text_imputer = use_cross_modal_text_imputer
        self.text_imputer_hidden_dim = text_imputer_hidden_dim
        self.text_imputer_residual_weight = text_imputer_residual_weight
        self.text_imputer_rec_weight = text_imputer_rec_weight
        self.text_imputer_nce_weight = text_imputer_nce_weight
        self.text_imputer_temperature = text_imputer_temperature
        self.cross_modal_text_imputer_effective = (
            bool(use_cross_modal_text_imputer) and
            (
                float(text_imputer_residual_weight) > 0.0 or
                float(text_imputer_rec_weight) > 0.0 or
                float(text_imputer_nce_weight) > 0.0
            )
        )
        self.use_confidence_gated_retrieval = use_confidence_gated_retrieval
        self.retrieval_gate_type = retrieval_gate_type
        self.retrieval_gate_min = retrieval_gate_min
        self.retrieval_gate_max = retrieval_gate_max
        self.use_relation_aware_retrieval = use_relation_aware_retrieval
        self.relation_head_text_pools = (
            [pool.detach().cpu().long() for pool in relation_head_text_pools]
            if relation_head_text_pools is not None else None
        )
        self.relation_tail_text_pools = (
            [pool.detach().cpu().long() for pool in relation_tail_text_pools]
            if relation_tail_text_pools is not None else None
        )
        self.min_relation_pool_size = min_relation_pool_size
        self.relation_retrieval_fallback = relation_retrieval_fallback
        if self.use_missingness_relation_expert:
            if self.disable_text:
                raise ValueError("Missingness-relation expert fusion requires the text branch to be enabled.")
            if self.expert_hidden_dim <= 0:
                raise ValueError("expert_hidden_dim must be > 0.")
            if self.expert_num != 2:
                raise ValueError("The minimal missingness-relation expert fusion supports exactly 2 experts.")
        if self.use_branch_local_relation_gate:
            if self.disable_text:
                raise ValueError("Branch-local relation gate requires the text branch to be enabled.")
            if self.branch_gate_hidden_dim <= 0:
                raise ValueError("branch_gate_hidden_dim must be > 0.")
            if not self.use_retrieval_missing_text:
                raise ValueError("Branch-local relation gate requires retrieval missing-text compensation.")
        if self.use_confidence_gated_retrieval and not self.use_retrieval_missing_text:
            raise ValueError("Confidence-gated retrieval requires retrieval missing-text to be enabled.")
        if self.use_retrieval_confidence_calibration and not self.use_retrieval_missing_text:
            raise ValueError("Retrieval confidence calibration requires retrieval missing-text to be enabled.")
        if self.use_retrieval_confidence_calibration and self.use_confidence_gated_retrieval:
            raise ValueError("Retrieval confidence calibration cannot be combined with confidence-gated retrieval.")
        if self.use_cross_modal_text_imputer:
            if self.disable_text:
                raise ValueError("Cross-modal text imputer requires the text branch to be enabled.")
            if not self.use_retrieval_missing_text:
                raise ValueError("Cross-modal text imputer requires retrieval missing-text compensation.")
            if self.text_imputer_hidden_dim <= 0:
                raise ValueError("text_imputer_hidden_dim must be > 0.")
            if self.text_imputer_residual_weight < 0.0:
                raise ValueError("text_imputer_residual_weight must be >= 0.0.")
            if self.text_imputer_rec_weight < 0.0:
                raise ValueError("text_imputer_rec_weight must be >= 0.0.")
            if self.text_imputer_nce_weight < 0.0:
                raise ValueError("text_imputer_nce_weight must be >= 0.0.")
            if self.text_imputer_temperature <= 0.0:
                raise ValueError("text_imputer_temperature must be > 0.")
            if self.use_retrieval_confidence_calibration:
                raise ValueError("Cross-modal text imputer should not be combined with retrieval confidence calibration.")
            if self.use_confidence_gated_retrieval:
                raise ValueError("Cross-modal text imputer should not be combined with confidence-gated retrieval.")
            if self.use_branch_local_relation_gate:
                raise ValueError("Cross-modal text imputer should not be combined with branch-local relation gate.")
            if self.use_missingness_relation_expert:
                raise ValueError("Cross-modal text imputer should not be combined with missingness-relation expert.")
            if self.use_relation_aware_retrieval:
                raise ValueError("Cross-modal text imputer keeps B-v1 retrieval and should not use relation-aware retrieval.")
            if use_missing_text_consistency:
                raise ValueError("Cross-modal text imputer should not be combined with missing-text consistency loss.")
        if self.use_retrieval_missing_text:
            if self.retrieval_source not in ("entity_embedding_knn", "random_text_pool"):
                raise ValueError(
                    "retrieval_source must be one of: 'entity_embedding_knn', 'random_text_pool'."
                )
            if self.retrieval_topk <= 0:
                raise ValueError("retrieval_topk must be > 0.")
            if self.retrieval_pool_size < 0:
                raise ValueError("retrieval_pool_size must be >= 0.")
            if self.retrieval_mix_weight < 0.0:
                raise ValueError("retrieval_mix_weight must be >= 0.0.")
            if self.use_retrieval_confidence_calibration:
                if self.retrieval_confidence_type not in ("mean_topk_similarity", "normalized_mean_topk_similarity"):
                    raise ValueError(
                        "retrieval_confidence_type must be one of: "
                        "'mean_topk_similarity', 'normalized_mean_topk_similarity'."
                    )
                if not 0.0 <= self.retrieval_confidence_min <= 1.0:
                    raise ValueError("retrieval_confidence_min must be in [0, 1].")
                if not 0.0 <= self.retrieval_confidence_max <= 1.0:
                    raise ValueError("retrieval_confidence_max must be in [0, 1].")
                if self.retrieval_confidence_min > self.retrieval_confidence_max:
                    raise ValueError("retrieval_confidence_min must be <= retrieval_confidence_max.")
            if self.use_confidence_gated_retrieval:
                if self.retrieval_gate_type != "similarity_based":
                    raise ValueError("Only retrieval_gate_type='similarity_based' is supported.")
                if not 0.0 <= self.retrieval_gate_min <= 1.0:
                    raise ValueError("retrieval_gate_min must be in [0, 1].")
                if not 0.0 <= self.retrieval_gate_max <= 1.0:
                    raise ValueError("retrieval_gate_max must be in [0, 1].")
                if self.retrieval_gate_min > self.retrieval_gate_max:
                    raise ValueError("retrieval_gate_min must be <= retrieval_gate_max.")
            if self.use_relation_aware_retrieval:
                if self.relation_retrieval_fallback != "global_text_pool":
                    raise ValueError("Only relation_retrieval_fallback='global_text_pool' is supported.")
                if self.min_relation_pool_size <= 0:
                    raise ValueError("min_relation_pool_size must be > 0.")
                if self.relation_head_text_pools is None or self.relation_tail_text_pools is None:
                    raise ValueError("Relation-aware retrieval requires relation head/tail text candidate pools.")
        self.entity_specific_missing_text_recon_weight = entity_specific_missing_text_recon_weight
        self.use_missing_text_consistency = use_missing_text_consistency
        self.consistency_prob = consistency_prob
        self.consistency_lambda = consistency_lambda
        self.use_missing_text_token_scale = use_missing_text_token_scale
        self.missing_image_token = nn.Parameter(torch.zeros(self.dim_e))
        self.missing_text_token = nn.Parameter(torch.zeros(self.dim_e))
        self.missing_text_token_head = nn.Parameter(torch.zeros(self.dim_e))
        self.missing_text_token_tail = nn.Parameter(torch.zeros(self.dim_e))
        self.missing_text_token_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.prototype_missing_text_cluster_ids = None
        self.prototype_missing_text_num_clusters = None
        if self.use_prototype_missing_text:
            if prototype_missing_text_cluster_ids is None or prototype_missing_text_num_clusters is None:
                raise ValueError("Prototype missing-text requires both cluster ids and num_clusters.")
            cluster_ids = torch.as_tensor(prototype_missing_text_cluster_ids, dtype=torch.long).view(-1).cpu()
            if cluster_ids.shape[0] != self.ent_tot:
                raise ValueError(
                    f"Prototype missing-text cluster count mismatch: expected {self.ent_tot}, got {cluster_ids.shape[0]}."
                )
            if bool((cluster_ids < 0).any().item()):
                raise ValueError("Prototype missing-text cluster ids must be non-negative.")
            self.prototype_missing_text_num_clusters = int(prototype_missing_text_num_clusters)
            if self.prototype_missing_text_num_clusters <= 0:
                raise ValueError("Prototype missing-text requires num_clusters > 0.")
            if cluster_ids.numel() > 0 and int(cluster_ids.max().item()) >= self.prototype_missing_text_num_clusters:
                raise ValueError(
                    "Prototype missing-text cluster ids exceed the declared num_clusters."
                )
            self.prototype_missing_text_cluster_ids = cluster_ids
            self.prototype_missing_text_token_bank = nn.Embedding(
                self.prototype_missing_text_num_clusters,
                self.dim_e,
            )
            nn.init.zeros_(self.prototype_missing_text_token_bank.weight)
        else:
            self.prototype_missing_text_token_bank = None
        self.entity_specific_missing_text_predictor = nn.Sequential(
            nn.Linear(self.dim_e * 2 + 1, self.dim_e),
            nn.GELU(),
            nn.Linear(self.dim_e, self.dim_e),
        )
        self.structure_conditioned_missing_text_predictor = nn.Sequential(
            nn.Linear(self.dim_e, self.dim_e),
            nn.GELU(),
            nn.Linear(self.dim_e, self.dim_e),
        )
        # Representation-level expert fusion modules are initialized after the
        # legacy embeddings so enabling A does not shift the baseline RNG stream.
        self.missingness_relation_expert_gate = None
        self.shared_representation_expert = None
        self.text_missing_representation_expert = None
        self.branch_local_relation_gate = None
        self.cross_modal_text_imputer = None
        self.cross_modal_text_imputer_residual_norm = None
        self.missing_text_gate_logit = nn.Parameter(torch.tensor([2.1972246], dtype=torch.float32))
        self.text_reliability_scorer = nn.Linear(4, 1)
        self.image_reliability_scorer = nn.Linear(4, 1)
        nn.init.zeros_(self.text_reliability_scorer.weight)
        nn.init.zeros_(self.text_reliability_scorer.bias)
        nn.init.zeros_(self.image_reliability_scorer.weight)
        nn.init.zeros_(self.image_reliability_scorer.bias)
        self.last_mask_debug = None
        self.full_text_score_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self.missing_text_score_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self._joint_scoring_combine_debug_printed = False
        self._joint_scoring_forward_debug_printed = False
        self._joint_scoring_negative_debug_printed = False
        self._missing_aware_joint_debug_counts = {}
        self._fusion_debug_printed = False
        self._missingness_relation_expert_debug_printed = False
        self._active_modality_log_phases = set()
        self._text_off_debug_printed = False
        self._pseudo_missing_debug_count = 0
        self._soft_missing_image_debug_count = 0
        self._generator_text_alignment_debug_printed = False
        self._entity_specific_missing_text_debug_count = 0
        self._last_entity_specific_missing_text_stats = None
        self._retrieval_missing_text_debug_count = 0
        self._retrieval_missing_text_debug_counts_by_context = {}
        self._last_retrieval_missing_text_stats = None
        self._retrieval_missing_text_stats_by_context = {}
        self._last_retrieval_confidence_calibration_stats = None
        self._last_cross_modal_text_imputer_stats = None
        self._last_missingness_relation_expert_stats = None
        self._branch_local_relation_gate_debug_printed = False
        self._last_branch_local_relation_gate_stats = None
        self._prototype_missing_text_debug_count = 0
        self._last_prototype_missing_text_stats = None
        self._structure_conditioned_missing_text_debug_count = 0
        self._last_structure_conditioned_missing_text_stats = None
        self._missing_text_consistency_debug_count = 0
        self._last_missing_text_consistency_stats = None
        self._pending_auxiliary_loss = None
        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if self.use_missingness_relation_expert:
            self.missingness_relation_expert_gate = nn.Sequential(
                nn.Linear(self.dim_r + 1, self.expert_hidden_dim),
                nn.GELU(),
                nn.Linear(self.expert_hidden_dim, self.expert_num),
            )
            self.shared_representation_expert = nn.Sequential(
                nn.Linear(self.dim_e, self.expert_hidden_dim),
                nn.GELU(),
                nn.Linear(self.expert_hidden_dim, self.dim_e),
            )
            self.text_missing_representation_expert = nn.Sequential(
                nn.Linear(self.dim_e, self.expert_hidden_dim),
                nn.GELU(),
                nn.Linear(self.expert_hidden_dim, self.dim_e),
            )
            nn.init.zeros_(self.shared_representation_expert[-1].weight)
            nn.init.zeros_(self.shared_representation_expert[-1].bias)
            nn.init.zeros_(self.text_missing_representation_expert[-1].weight)
            nn.init.zeros_(self.text_missing_representation_expert[-1].bias)
        if self.use_branch_local_relation_gate:
            self.branch_local_relation_gate = nn.Sequential(
                nn.Linear(self.dim_r + 1, self.branch_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.branch_gate_hidden_dim, 1),
            )
        if self.cross_modal_text_imputer_effective:
            # RG-CTD is a missing-text compensation residual, not a score router.
            rng_state = torch.get_rng_state()
            try:
                self.cross_modal_text_imputer = nn.Sequential(
                    nn.Linear(self.dim_e * 5, self.text_imputer_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.text_imputer_hidden_dim, self.dim_e),
                    nn.LayerNorm(self.dim_e),
                )
                self.cross_modal_text_imputer_residual_norm = nn.LayerNorm(self.dim_e)
                nn.init.zeros_(self.cross_modal_text_imputer[3].weight)
                nn.init.zeros_(self.cross_modal_text_imputer[3].bias)
            finally:
                # Optional RG-CTD parameters should not shift the B-v1 RNG stream
                # used by later modules such as DiffHEG.
                torch.set_rng_state(rng_state)

    def uses_text_branch(self):
        return not self.disable_text

    def get_active_modalities(self):
        modalities = ["structural", "visual"]
        if self.uses_text_branch():
            modalities.append("text")
        return modalities

    def get_conditioning_availability_dim(self):
        return 4 if self.uses_text_branch() else 2

    def get_text_branch_debug_state(self):
        return {
            "text_branch_enabled": self.uses_text_branch(),
            "active_modalities": self.get_active_modalities(),
            "active_modality_count": len(self.get_active_modalities()),
            "text_off_debug_printed": self._text_off_debug_printed,
            "pseudo_missing_prob": self.pseudo_missing_prob,
            "pseudo_missing_debug_count": self._pseudo_missing_debug_count,
            "use_fixed_zero_missing_text": self.use_fixed_zero_missing_text,
            "use_side_aware_missing_text": self.use_side_aware_missing_text,
            "use_soft_token_text_generator_alignment": self.use_soft_token_text_generator_alignment,
            "generator_text_alignment_debug_printed": self._generator_text_alignment_debug_printed,
            "use_missingness_relation_expert": self.use_missingness_relation_expert,
            "expert_hidden_dim": self.expert_hidden_dim,
            "expert_num": self.expert_num,
            "last_missingness_relation_expert_stats": self._last_missingness_relation_expert_stats,
            "use_branch_local_relation_gate": self.use_branch_local_relation_gate,
            "branch_gate_hidden_dim": self.branch_gate_hidden_dim,
            "last_branch_local_relation_gate_stats": self._last_branch_local_relation_gate_stats,
            "use_entity_specific_missing_text": self.use_entity_specific_missing_text,
            "use_retrieval_missing_text": self.use_retrieval_missing_text,
            "retrieval_topk": self.retrieval_topk,
            "retrieval_pool_size": self.retrieval_pool_size,
            "retrieval_source": self.retrieval_source,
            "retrieval_mix_weight": self.retrieval_mix_weight,
            "use_retrieval_confidence_calibration": self.use_retrieval_confidence_calibration,
            "retrieval_confidence_type": self.retrieval_confidence_type,
            "retrieval_confidence_min_config": self.retrieval_confidence_min,
            "retrieval_confidence_max_config": self.retrieval_confidence_max,
            "last_retrieval_confidence_calibration_stats": self._last_retrieval_confidence_calibration_stats,
            "use_cross_modal_text_imputer": self.use_cross_modal_text_imputer,
            "cross_modal_text_imputer_effective": self.cross_modal_text_imputer_effective,
            "text_imputer_hidden_dim": self.text_imputer_hidden_dim,
            "text_imputer_residual_weight": self.text_imputer_residual_weight,
            "text_imputer_rec_weight": self.text_imputer_rec_weight,
            "text_imputer_nce_weight": self.text_imputer_nce_weight,
            "text_imputer_temperature": self.text_imputer_temperature,
            "last_cross_modal_text_imputer_stats": self._last_cross_modal_text_imputer_stats,
            "use_confidence_gated_retrieval": self.use_confidence_gated_retrieval,
            "retrieval_gate_type": self.retrieval_gate_type,
            "retrieval_gate_min": self.retrieval_gate_min,
            "retrieval_gate_max": self.retrieval_gate_max,
            "use_relation_aware_retrieval": self.use_relation_aware_retrieval,
            "min_relation_pool_size": self.min_relation_pool_size,
            "relation_retrieval_fallback": self.relation_retrieval_fallback,
            "retrieval_missing_text_debug_count": self._retrieval_missing_text_debug_count,
            "retrieval_missing_text_debug_counts_by_context": dict(self._retrieval_missing_text_debug_counts_by_context),
            "last_retrieval_missing_text_stats": self._last_retrieval_missing_text_stats,
            "retrieval_missing_text_stats_by_context": self.get_relation_aware_retrieval_stats(),
            "entity_specific_missing_text_recon_weight": self.entity_specific_missing_text_recon_weight,
            "entity_specific_missing_text_debug_count": self._entity_specific_missing_text_debug_count,
            "last_entity_specific_missing_text_stats": self._last_entity_specific_missing_text_stats,
            "use_prototype_missing_text": self.use_prototype_missing_text,
            "prototype_missing_text_num_clusters": self.prototype_missing_text_num_clusters,
            "prototype_missing_text_debug_count": self._prototype_missing_text_debug_count,
            "last_prototype_missing_text_stats": self._last_prototype_missing_text_stats,
            "use_structure_conditioned_missing_text": self.use_structure_conditioned_missing_text,
            "structure_conditioned_missing_text_debug_count": self._structure_conditioned_missing_text_debug_count,
            "last_structure_conditioned_missing_text_stats": self._last_structure_conditioned_missing_text_stats,
            "use_missing_text_consistency": self.use_missing_text_consistency,
            "consistency_prob": self.consistency_prob,
            "consistency_lambda": self.consistency_lambda,
            "missing_text_consistency_debug_count": self._missing_text_consistency_debug_count,
            "last_missing_text_consistency_stats": self._last_missing_text_consistency_stats,
            "use_missing_text_token_scale": self.use_missing_text_token_scale,
            "missing_text_token_norm": float(self.missing_text_token.detach().norm().item()),
            "missing_text_token_head_norm": float(self.missing_text_token_head.detach().norm().item()),
            "missing_text_token_tail_norm": float(self.missing_text_token_tail.detach().norm().item()),
            "missing_text_token_scale": float(self.missing_text_token_scale.detach().item()),
        }

    def get_missingness_relation_expert_state(self):
        return {
            "enabled": bool(self.use_missingness_relation_expert),
            "expert_hidden_dim": self.expert_hidden_dim,
            "expert_num": self.expert_num,
            "last_stats": self._last_missingness_relation_expert_stats,
        }

    def get_branch_local_relation_gate_state(self):
        return {
            "enabled": bool(self.use_branch_local_relation_gate),
            "branch_gate_hidden_dim": self.branch_gate_hidden_dim,
            "last_stats": self._last_branch_local_relation_gate_stats,
        }

    def _maybe_log_active_modalities(self, phase):
        if phase in self._active_modality_log_phases:
            return
        modalities = self.get_active_modalities()
        print(
            "Active modalities | phase=%s | text_branch_enabled=%s | modality_count=%d | modalities=%s"
            % (
                phase,
                "True" if self.uses_text_branch() else "False",
                len(modalities),
                ",".join(modalities),
            )
        )
        self._active_modality_log_phases.add(phase)

    def _maybe_log_text_branch_skipped(self, context):
        if self.uses_text_branch() or self._text_off_debug_printed:
            return
        print(
            "Text-off debug | context=%s | text_embeddings=None | text_branch_skipped=True | oracle_restore_skipped=True | soft_missing_text_skipped=True"
            % context
        )
        self._text_off_debug_printed = True

    def _maybe_log_generator_text_alignment(self, applied_soft_missing):
        if self._generator_text_alignment_debug_printed:
            return
        print(
            "Text generator alignment | enabled=True | use_soft_missing_text=%s | applied_soft_missing_path=%s | context=get_batch_text_embs"
            % (
                "True" if self.use_soft_missing_text else "False",
                "True" if applied_soft_missing else "False",
            )
        )
        self._generator_text_alignment_debug_printed = True

    def _infer_retrieval_phase(self, label, phase=None):
        if phase in ("train", "eval"):
            return phase
        if isinstance(label, str) and label.startswith("eval"):
            return "eval"
        return "train"

    def _make_text_context(self, label, relation_ids=None, retrieval_role=None, phase=None):
        if (
            (self.use_relation_aware_retrieval or self.use_branch_local_relation_gate) and
            relation_ids is not None and
            retrieval_role is not None
        ):
            if not torch.is_tensor(relation_ids):
                return label
            if relation_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                return label
            return {
                "label": label,
                "relation_ids": relation_ids,
                "retrieval_role": retrieval_role,
                "retrieval_phase": self._infer_retrieval_phase(label, phase=phase),
            }
        return label

    def _get_text_branch_embeddings(self, batch_entities, context, allow_pseudo_missing=True):
        if not self.uses_text_branch():
            self._maybe_log_text_branch_skipped(context)
            return None
        text_emb = self.text_proj(self.text_embeddings(batch_entities))
        text_emb = self._apply_oracle_restore_text(text_emb, batch_entities)
        text_emb = self._apply_fixed_zero_missing_text(text_emb, batch_entities)
        return self._apply_soft_missing_text(
            text_emb,
            batch_entities,
            allow_pseudo_missing=allow_pseudo_missing,
            context=context,
        )

    def _apply_soft_missing_image(self, image_emb, batch_entities):
        if not self.use_soft_missing_image or self.has_image is None:
            return image_emb
        image_mask = self._get_entity_mask(batch_entities, self.has_image, image_emb.device)
        if image_mask is None or bool(image_mask.all().item()):
            return image_emb
        missing_token = self.missing_image_token.unsqueeze(0).expand_as(image_emb)
        output_image = torch.where(image_mask, image_emb, missing_token)
        observed_mask = image_mask.expand_as(image_emb)
        missing_mask = ~observed_mask
        if bool(observed_mask.any().item()):
            assert torch.equal(output_image[observed_mask], image_emb[observed_mask]), (
                "soft missing-image should not alter observed image embeddings"
            )
        if bool(missing_mask.any().item()):
            assert torch.equal(output_image[missing_mask], missing_token[missing_mask]), (
                "soft missing-image should only replace has_image=False positions with missing_image_token"
            )
            if self._soft_missing_image_debug_count < 3:
                print(
                    "Soft missing-image debug | missing_count=%d | observed_count=%d | replacement_scope=has_image_false_only"
                    % (
                        int((~image_mask).sum().item()),
                        int(image_mask.sum().item()),
                    )
                )
                self._soft_missing_image_debug_count += 1
        return output_image

    def _get_image_branch_embeddings(self, batch_entities):
        image_emb = self.img_proj(self.img_embeddings(batch_entities))
        return self._apply_soft_missing_image(image_emb, batch_entities)

    def _get_image_availability(self, batch_entities, device):
        return self._get_entity_availability(batch_entities, self.has_image, device)

    def _predict_entity_specific_missing_text(self, batch_entities):
        struct_emb = self.ent_embeddings(batch_entities)
        image_emb = self.img_proj(self.img_embeddings(batch_entities))
        image_available = self._get_image_availability(batch_entities, struct_emb.device)
        fused_image = image_emb * image_available
        predictor_input = torch.cat([struct_emb, fused_image, image_available], dim=-1)
        return self.entity_specific_missing_text_predictor(predictor_input)

    def _predict_structure_conditioned_missing_text(self, batch_entities):
        struct_emb = self.ent_embeddings(batch_entities)
        return self.structure_conditioned_missing_text_predictor(struct_emb)

    def _get_entity_mask(self, batch_entities, entity_mask, device):
        if entity_mask is None:
            return None
        entity_ids = batch_entities.detach().view(-1).cpu()
        return entity_mask.index_select(0, entity_ids).to(device=device, dtype=torch.bool).unsqueeze(-1)

    def _should_use_text_mask(self):
        return self.uses_text_branch() and self.use_missing_mask and not self.use_soft_missing_text

    def _get_oracle_restore_mask(self, batch_entities, device):
        if not self.use_oracle_restore_injected_text or self.injected_text_mask is None:
            return None
        return self._get_entity_mask(batch_entities, self.injected_text_mask, device)

    def _resolve_missing_text_side(self, context=None):
        if context is None:
            return None
        if isinstance(context, dict):
            if context.get("retrieval_role") in ("head", "tail"):
                return context["retrieval_role"]
            context = context.get("label")
            if context is None:
                return None
        if context.endswith("_head"):
            return "head"
        if context.endswith("_tail"):
            return "tail"
        return None

    def _apply_oracle_restore_text(self, text_emb, batch_entities):
        if not self.uses_text_branch() or not self.use_oracle_restore_injected_text or self.oracle_text_embeddings is None:
            return text_emb
        oracle_restore_mask = self._get_oracle_restore_mask(batch_entities, text_emb.device)
        if oracle_restore_mask is None or not bool(oracle_restore_mask.any().item()):
            return text_emb
        oracle_text_emb = self.text_proj(self.oracle_text_embeddings(batch_entities))
        return torch.where(oracle_restore_mask, oracle_text_emb, text_emb)

    def _apply_fixed_zero_missing_text(self, text_emb, batch_entities):
        if not self.uses_text_branch() or not self.use_fixed_zero_missing_text or self.has_text is None:
            return text_emb
        text_mask = self._get_entity_mask(batch_entities, self.has_text, text_emb.device)
        oracle_restore_mask = self._get_oracle_restore_mask(batch_entities, text_emb.device)
        if oracle_restore_mask is not None:
            text_mask = text_mask | oracle_restore_mask
        if text_mask is None or bool(text_mask.all().item()):
            return text_emb
        return torch.where(text_mask, text_emb, torch.zeros_like(text_emb))

    def _get_missing_text_token_value(self, text_emb, context=None):
        side = self._resolve_missing_text_side(context) if self.use_side_aware_missing_text else None
        if side == "head":
            missing_token = self.missing_text_token_head
        elif side == "tail":
            missing_token = self.missing_text_token_tail
        else:
            missing_token = self.missing_text_token
        if self.use_missing_text_token_scale:
            missing_token = missing_token * self.missing_text_token_scale.to(
                device=missing_token.device,
                dtype=missing_token.dtype,
            )
        return missing_token.to(device=text_emb.device, dtype=text_emb.dtype)

    def _limit_retrieval_pool(self, candidate_ids, device):
        candidate_ids = candidate_ids.to(device=device, dtype=torch.long)
        if self.retrieval_pool_size > 0 and candidate_ids.numel() > self.retrieval_pool_size:
            # Lightweight probe path: keep a deterministic cap before KNN.
            pool_positions = torch.linspace(
                0,
                candidate_ids.numel() - 1,
                steps=self.retrieval_pool_size,
                device=device,
            ).round().to(dtype=torch.long)
            candidate_ids = candidate_ids.index_select(0, pool_positions)
        return candidate_ids

    def _get_global_text_retrieval_pool(self, device):
        available_ids = torch.nonzero(self.has_text, as_tuple=False).view(-1)
        full_candidate_count = int(available_ids.numel())
        if full_candidate_count == 0:
            return available_ids.to(device=device, dtype=torch.long), full_candidate_count
        return self._limit_retrieval_pool(available_ids, device), full_candidate_count

    def _expand_relation_ids_for_retrieval(self, relation_ids, flat_count, device):
        if relation_ids is None:
            return None
        relation_ids = relation_ids.detach().view(-1).to(device=device, dtype=torch.long)
        if relation_ids.numel() == flat_count:
            return relation_ids
        if relation_ids.numel() == 1:
            return relation_ids.expand(flat_count)
        if flat_count % relation_ids.numel() == 0:
            # RotatE scoring reshapes candidates as [candidate_slot, relation_batch].
            return relation_ids.repeat(flat_count // relation_ids.numel())
        return None

    def _get_retrieval_context(self, context, flat_count, device):
        if not isinstance(context, dict):
            return None, None, None
        role = context.get("retrieval_role")
        if role not in ("head", "tail"):
            return None, None, None
        relation_ids = self._expand_relation_ids_for_retrieval(
            context.get("relation_ids"),
            flat_count,
            device,
        )
        if relation_ids is None:
            return None, None, None
        phase = context.get("retrieval_phase")
        if phase not in ("train", "eval"):
            phase = self._infer_retrieval_phase(context.get("label"))
        return relation_ids, role, phase

    def _get_relation_text_pool(self, relation_id, role):
        pools = self.relation_head_text_pools if role == "head" else self.relation_tail_text_pools
        if pools is None or relation_id < 0 or relation_id >= len(pools):
            return None
        return pools[relation_id]

    def _knn_text_aggregation_from_pool(self, flat_entities, candidate_ids, text_dtype):
        topk = min(int(self.retrieval_topk), int(candidate_ids.numel()))
        if topk <= 0:
            empty_stats = {
                "topk_similarity_mean": torch.zeros(flat_entities.shape[0], device=flat_entities.device),
                "topk_similarity_max": torch.zeros(flat_entities.shape[0], device=flat_entities.device),
                "topk_similarity_std": torch.zeros(flat_entities.shape[0], device=flat_entities.device),
            }
            return (
                torch.zeros(flat_entities.shape[0], self.dim_e, device=flat_entities.device, dtype=text_dtype),
                empty_stats,
            )
        with torch.no_grad():
            query_struct = F.normalize(
                self.ent_embeddings(flat_entities).detach(),
                p=2,
                dim=-1,
                eps=1e-12,
            )
            candidate_struct = F.normalize(
                self.ent_embeddings.weight.index_select(0, candidate_ids).detach(),
                p=2,
                dim=-1,
                eps=1e-12,
            )
            similarity = torch.matmul(query_struct, candidate_struct.transpose(0, 1))
            if self.retrieval_source == "entity_embedding_knn":
                topk_scores, topk_positions = torch.topk(similarity, k=topk, dim=-1)
                retrieval_weights = torch.softmax(topk_scores, dim=-1)
            elif self.retrieval_source == "random_text_pool":
                random_scores = self._deterministic_random_retrieval_scores(flat_entities, candidate_ids)
                _unused_scores, topk_positions = torch.topk(random_scores, k=topk, dim=-1)
                topk_scores = torch.gather(similarity, dim=1, index=topk_positions)
                retrieval_weights = torch.full_like(topk_scores, 1.0 / float(topk))
            else:
                raise ValueError(
                    "retrieval_source must be one of: 'entity_embedding_knn', 'random_text_pool'."
                )
            retrieved_ids = candidate_ids.index_select(0, topk_positions.reshape(-1)).view(flat_entities.shape[0], topk)
            retrieval_stats = {
                "topk_similarity_mean": topk_scores.mean(dim=-1).detach().to(dtype=torch.float32),
                "topk_similarity_max": topk_scores.max(dim=-1).values.detach().to(dtype=torch.float32),
                "topk_similarity_std": topk_scores.std(dim=-1, unbiased=False).detach().to(dtype=torch.float32),
            }

        prototype_text = self.text_proj(self.text_embeddings(retrieved_ids.reshape(-1)))
        prototype_text = prototype_text.view(flat_entities.shape[0], topk, self.dim_e)
        prototype_text_agg = (prototype_text * retrieval_weights.unsqueeze(-1).to(dtype=prototype_text.dtype)).sum(dim=1)
        return prototype_text_agg.to(dtype=text_dtype), retrieval_stats

    def _deterministic_random_retrieval_scores(self, flat_entities, candidate_ids):
        entity_key = flat_entities.to(dtype=torch.float32).view(-1, 1) + 1.0
        candidate_key = candidate_ids.to(device=flat_entities.device, dtype=torch.float32).view(1, -1) + 1.0
        scores = torch.sin(entity_key * 12.9898 + candidate_key * 78.233) * 43758.5453
        return scores - torch.floor(scores)

    def _merge_cross_modal_text_imputer_stats(self, stats):
        merged_stats = dict(self._last_cross_modal_text_imputer_stats or {})
        merged_stats.update(stats)
        self._last_cross_modal_text_imputer_stats = merged_stats

    def _predict_cross_modal_pseudo_text(self, batch_entities, prototype_text_agg, detach_inputs=False):
        if not self.use_cross_modal_text_imputer or self.cross_modal_text_imputer is None:
            return prototype_text_agg
        original_shape = prototype_text_agg.shape
        flat_entities = batch_entities.detach().view(-1).to(
            device=prototype_text_agg.device,
            dtype=torch.long,
        )
        flat_prototype = prototype_text_agg.reshape(-1, self.dim_e)
        structural_emb = self.ent_embeddings(flat_entities)
        visual_emb = self._get_image_branch_embeddings(flat_entities)
        if detach_inputs:
            structural_emb = structural_emb.detach()
            visual_emb = visual_emb.detach()
            flat_prototype = flat_prototype.detach()
        imputer_input = torch.cat(
            [
                structural_emb,
                visual_emb,
                flat_prototype,
                structural_emb * flat_prototype,
                visual_emb * flat_prototype,
            ],
            dim=-1,
        )
        # The imputer predicts a zero-initialized delta around the B-v1 prototype,
        # so RG-CTD starts as the original retrieval compensation path.
        pseudo_text = flat_prototype + self.cross_modal_text_imputer(imputer_input)
        return pseudo_text.view(*original_shape)

    def _apply_cross_modal_text_imputer_residual(
        self,
        batch_entities,
        retrieval_augmented_text,
        prototype_text_agg,
        replacement_mask=None,
    ):
        if (
            not self.use_cross_modal_text_imputer or
            self.cross_modal_text_imputer is None or
            self.cross_modal_text_imputer_residual_norm is None or
            self.text_imputer_residual_weight <= 0.0
        ):
            return retrieval_augmented_text
        pseudo_text = self._predict_cross_modal_pseudo_text(batch_entities, prototype_text_agg)
        residual = self.cross_modal_text_imputer_residual_norm(pseudo_text - prototype_text_agg)
        compensated_text = retrieval_augmented_text + self.text_imputer_residual_weight * residual

        flat_residual = residual.detach().reshape(-1, self.dim_e)
        flat_pseudo = pseudo_text.detach().reshape(-1, self.dim_e)
        flat_prototype = prototype_text_agg.detach().reshape(-1, self.dim_e)
        flat_compensated = compensated_text.detach().reshape(-1, self.dim_e)
        if replacement_mask is None:
            stats_mask = torch.ones(flat_residual.shape[0], device=flat_residual.device, dtype=torch.bool)
        else:
            stats_mask = replacement_mask.detach().view(-1).to(device=flat_residual.device, dtype=torch.bool)

        def _masked_norm_mean(tensor):
            if not bool(stats_mask.any().item()):
                return None
            return float(tensor.norm(dim=-1).masked_select(stats_mask).float().mean().item())

        self._merge_cross_modal_text_imputer_stats(
            {
                "text_imputer_residual_norm_mean": _masked_norm_mean(flat_residual),
                "pseudo_text_norm_mean": _masked_norm_mean(flat_pseudo),
                "prototype_text_agg_mean_norm": _masked_norm_mean(flat_prototype),
                "missing_text_compensated_norm_mean": _masked_norm_mean(flat_compensated),
                "missing_text_entity_count": int(stats_mask.sum().item()),
            }
        )
        return compensated_text

    def _compute_cross_modal_text_imputer_loss(self, batch_entities):
        if (
            not self.training or
            not self.use_cross_modal_text_imputer or
            self.cross_modal_text_imputer is None or
            self.has_text is None or
            (self.text_imputer_rec_weight <= 0.0 and self.text_imputer_nce_weight <= 0.0)
        ):
            return None
        unique_entities = torch.unique(batch_entities.detach().view(-1)).to(
            device=self.ent_embeddings.weight.device,
            dtype=torch.long,
        )
        if unique_entities.numel() == 0:
            return None
        supervision_mask = self._get_text_reconstruction_mask(unique_entities, self.ent_embeddings.weight.device)
        if supervision_mask is None:
            return None
        supervision_mask = supervision_mask.squeeze(-1)
        if not bool(supervision_mask.any().item()):
            self._merge_cross_modal_text_imputer_stats(
                {
                    "text_imputer_rec_loss": None,
                    "text_imputer_nce_loss": None,
                    "text_imputer_supervised_entity_count": 0,
                }
            )
            return None

        supervised_entities = unique_entities[supervision_mask]
        target_text = self.text_proj(self.text_embeddings(supervised_entities))
        target_text = self._apply_oracle_restore_text(target_text, supervised_entities).detach()
        global_ids, _ = self._get_global_text_retrieval_pool(target_text.device)
        if global_ids.numel() == 0:
            return None
        with torch.no_grad():
            prototype_text_agg, _ = self._knn_text_aggregation_from_pool(
                supervised_entities,
                global_ids,
                target_text.dtype,
            )
        pseudo_text = self._predict_cross_modal_pseudo_text(
            supervised_entities,
            prototype_text_agg,
            detach_inputs=True,
        )
        rec_loss = 1.0 - F.cosine_similarity(pseudo_text, target_text, dim=-1, eps=1e-8).mean()
        logits = torch.matmul(
            F.normalize(pseudo_text, p=2, dim=-1, eps=1e-12),
            F.normalize(target_text, p=2, dim=-1, eps=1e-12).transpose(0, 1),
        ) / float(self.text_imputer_temperature)
        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
        nce_loss = F.cross_entropy(logits, labels)
        self._merge_cross_modal_text_imputer_stats(
            {
                "text_imputer_rec_loss": float(rec_loss.detach().item()),
                "text_imputer_nce_loss": float(nce_loss.detach().item()),
                "text_imputer_supervised_entity_count": int(supervised_entities.numel()),
            }
        )
        return self.text_imputer_rec_weight * rec_loss + self.text_imputer_nce_weight * nce_loss

    def _build_confidence_retrieval_gate(self, retrieval_stats, text_emb):
        if not self.use_confidence_gated_retrieval:
            return None
        if self.retrieval_gate_type != "similarity_based":
            raise ValueError("Only retrieval_gate_type='similarity_based' is supported.")
        topk_similarity_mean = retrieval_stats.get("topk_similarity_mean")
        if topk_similarity_mean is None:
            return None
        confidence = ((topk_similarity_mean.to(device=text_emb.device, dtype=torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)
        gate = self.retrieval_gate_min + (self.retrieval_gate_max - self.retrieval_gate_min) * confidence
        return gate.to(device=text_emb.device, dtype=text_emb.dtype)

    def _summarize_retrieval_gate(self, retrieval_gate, replacement_mask):
        missing_count = 0
        replacement_mask_flat = None
        if replacement_mask is not None:
            replacement_mask_flat = replacement_mask.detach().view(-1).to(dtype=torch.bool)
            missing_count = int(replacement_mask_flat.sum().item())
        if retrieval_gate is None:
            return None, None, missing_count
        gate_flat = retrieval_gate.detach().view(-1).float()
        gate_mean = float(gate_flat.mean().item()) if gate_flat.numel() > 0 else None
        if replacement_mask_flat is None:
            return gate_mean, None, 0
        replacement_mask_flat = replacement_mask_flat.to(device=gate_flat.device)
        missing_gate_mean = (
            float(gate_flat.masked_select(replacement_mask_flat).mean().item())
            if missing_count > 0 else None
        )
        return gate_mean, missing_gate_mean, missing_count

    def _update_retrieval_missing_text_stats(self, stats):
        phase = stats.get("phase") or "unknown"
        role = stats.get("relation_role") or "unknown"
        key = f"{phase}:{role}"
        record = self._retrieval_missing_text_stats_by_context.setdefault(
            key,
            {
                "phase": phase,
                "relation_role": role,
                "calls": 0,
                "entity_count": 0,
                "fallback_entity_count": 0,
                "relation_pool_size_entity_sum": 0.0,
                "topk_similarity_mean_entity_sum": 0.0,
                "retrieval_gate_entity_sum": 0.0,
                "missing_text_entity_count": 0,
                "missing_text_gate_entity_sum": 0.0,
                "source_counts": {},
                "last_batch_stats": None,
            },
        )
        entity_count = int(stats.get("batch_entity_count", 0))
        record["calls"] += 1
        record["entity_count"] += entity_count
        record["fallback_entity_count"] += int(stats.get("fallback_entity_count", 0))
        record["relation_pool_size_entity_sum"] += float(stats.get("relation_pool_size_entity_sum", 0.0))
        topk_similarity_mean = stats.get("topk_similarity_mean_avg")
        if topk_similarity_mean is not None:
            record["topk_similarity_mean_entity_sum"] += float(topk_similarity_mean) * entity_count
        retrieval_gate_mean = stats.get("retrieval_gate_mean")
        if retrieval_gate_mean is not None:
            record["retrieval_gate_entity_sum"] += float(retrieval_gate_mean) * entity_count
        missing_text_count = int(stats.get("missing_text_entity_count", 0))
        record["missing_text_entity_count"] += missing_text_count
        missing_text_gate_mean = stats.get("missing_text_gate_mean")
        if missing_text_gate_mean is not None:
            record["missing_text_gate_entity_sum"] += float(missing_text_gate_mean) * missing_text_count
        for source, count in (stats.get("source_entity_counts") or {}).items():
            record["source_counts"][source] = int(record["source_counts"].get(source, 0)) + int(count)
        record["last_batch_stats"] = stats

    def get_relation_aware_retrieval_stats(self):
        summary = {}
        for key, record in sorted(self._retrieval_missing_text_stats_by_context.items()):
            entity_count = int(record.get("entity_count", 0))
            fallback_entity_count = int(record.get("fallback_entity_count", 0))
            source_counts = {
                source: int(count)
                for source, count in sorted((record.get("source_counts") or {}).items())
            }
            summary[key] = {
                "phase": record.get("phase"),
                "relation_role": record.get("relation_role"),
                "calls": int(record.get("calls", 0)),
                "entity_count": entity_count,
                "missing_text_entity_count": int(record.get("missing_text_entity_count", 0)),
                "avg_topk_similarity_mean": (
                    float(record.get("topk_similarity_mean_entity_sum", 0.0) / entity_count)
                    if entity_count > 0 else None
                ),
                "avg_retrieval_gate": (
                    float(record.get("retrieval_gate_entity_sum", 0.0) / entity_count)
                    if entity_count > 0 and self.use_confidence_gated_retrieval else None
                ),
                "avg_missing_text_gate": (
                    float(
                        record.get("missing_text_gate_entity_sum", 0.0)
                        / int(record.get("missing_text_entity_count", 0))
                    )
                    if int(record.get("missing_text_entity_count", 0)) > 0 and self.use_confidence_gated_retrieval else None
                ),
                "avg_relation_pool_size": (
                    float(record.get("relation_pool_size_entity_sum", 0.0) / entity_count)
                    if entity_count > 0 else None
                ),
                "fallback_ratio": (
                    float(fallback_entity_count / entity_count)
                    if entity_count > 0 else 0.0
                ),
                "retrieval_source": "+".join(source_counts.keys()) if source_counts else None,
                "source_counts": source_counts,
                "last_batch_stats": record.get("last_batch_stats"),
            }
        return summary

    def get_retrieval_missing_text_stats(self):
        return {
            "enabled": bool(self.use_retrieval_missing_text),
            "retrieval_source_config": self.retrieval_source,
            "use_confidence_gated_retrieval": bool(self.use_confidence_gated_retrieval),
            "retrieval_gate_type": self.retrieval_gate_type,
            "retrieval_gate_min": self.retrieval_gate_min,
            "retrieval_gate_max": self.retrieval_gate_max,
            "retrieval_confidence_calibration": self.get_retrieval_confidence_calibration_state(),
            "last_batch_stats": self._last_retrieval_missing_text_stats,
            "stats_by_context": self.get_relation_aware_retrieval_stats(),
        }

    def get_retrieval_confidence_calibration_state(self):
        return {
            "enabled": bool(self.use_retrieval_confidence_calibration),
            "retrieval_confidence_type": self.retrieval_confidence_type,
            "retrieval_confidence_min_config": self.retrieval_confidence_min,
            "retrieval_confidence_max_config": self.retrieval_confidence_max,
            "last_stats": self._last_retrieval_confidence_calibration_stats,
        }

    def get_cross_modal_text_imputer_state(self):
        return {
            "enabled": bool(self.use_cross_modal_text_imputer),
            "effective_enabled": bool(self.cross_modal_text_imputer_effective),
            "text_imputer_hidden_dim": self.text_imputer_hidden_dim,
            "text_imputer_residual_weight": self.text_imputer_residual_weight,
            "text_imputer_rec_weight": self.text_imputer_rec_weight,
            "text_imputer_nce_weight": self.text_imputer_nce_weight,
            "text_imputer_temperature": self.text_imputer_temperature,
            "last_stats": self._last_cross_modal_text_imputer_stats,
        }

    def _build_retrieval_confidence(self, retrieval_stats, text_emb):
        if not self.use_retrieval_confidence_calibration:
            return None
        if self.retrieval_confidence_type not in ("mean_topk_similarity", "normalized_mean_topk_similarity"):
            raise ValueError(
                "retrieval_confidence_type must be one of: "
                "'mean_topk_similarity', 'normalized_mean_topk_similarity'."
            )
        topk_similarity_mean = retrieval_stats.get("topk_similarity_mean")
        if topk_similarity_mean is None:
            return None
        confidence = topk_similarity_mean.to(device=text_emb.device, dtype=torch.float32)
        if self.retrieval_confidence_type == "normalized_mean_topk_similarity":
            confidence = (confidence + 1.0) / 2.0
        confidence = confidence.clamp(
            min=float(self.retrieval_confidence_min),
            max=float(self.retrieval_confidence_max),
        )
        return confidence.to(device=text_emb.device, dtype=text_emb.dtype)

    def _update_retrieval_confidence_calibration_stats(self, retrieval_confidence, prototype_text_agg, replacement_mask):
        if not self.use_retrieval_confidence_calibration:
            self._last_retrieval_confidence_calibration_stats = None
            return
        if retrieval_confidence is None:
            self._last_retrieval_confidence_calibration_stats = {
                "retrieval_confidence_mean": None,
                "retrieval_confidence_min": None,
                "retrieval_confidence_max": None,
                "calibrated_retrieval_norm_mean": None,
            }
            return
        confidence_flat = retrieval_confidence.detach().view(-1).float()
        calibrated_retrieval = prototype_text_agg * retrieval_confidence.view(-1, 1).to(
            device=prototype_text_agg.device,
            dtype=prototype_text_agg.dtype,
        )
        calibrated_norm = calibrated_retrieval.detach().norm(dim=-1).float()
        if replacement_mask is not None:
            missing_mask = replacement_mask.detach().view(-1).to(device=confidence_flat.device, dtype=torch.bool)
            confidence_for_stats = confidence_flat.masked_select(missing_mask)
            calibrated_norm_for_stats = calibrated_norm.masked_select(missing_mask)
        else:
            confidence_for_stats = confidence_flat
            calibrated_norm_for_stats = calibrated_norm
        self._last_retrieval_confidence_calibration_stats = {
            "retrieval_confidence_mean": (
                float(confidence_for_stats.mean().item())
                if confidence_for_stats.numel() > 0 else None
            ),
            "retrieval_confidence_min": (
                float(confidence_for_stats.min().item())
                if confidence_for_stats.numel() > 0 else None
            ),
            "retrieval_confidence_max": (
                float(confidence_for_stats.max().item())
                if confidence_for_stats.numel() > 0 else None
            ),
            "calibrated_retrieval_norm_mean": (
                float(calibrated_norm_for_stats.mean().item())
                if calibrated_norm_for_stats.numel() > 0 else None
            ),
        }

    def _get_retrieval_missing_text_aggregation(self, batch_entities, text_emb, context=None, replacement_mask=None):
        if not self.use_retrieval_missing_text:
            return torch.zeros_like(text_emb), None, None
        if self.has_text is None:
            return torch.zeros_like(text_emb), None, None
        if self.retrieval_source not in ("entity_embedding_knn", "random_text_pool"):
            raise ValueError(
                "retrieval_source must be one of: 'entity_embedding_knn', 'random_text_pool'."
            )

        device = text_emb.device
        global_ids, global_candidate_count = self._get_global_text_retrieval_pool(device)
        if global_ids.numel() == 0:
            return torch.zeros_like(text_emb), None, None
        flat_entities = batch_entities.detach().view(-1).to(device=device, dtype=torch.long)

        relation_ids, role, phase = self._get_retrieval_context(context, flat_entities.shape[0], device)
        use_relation_pool = self.use_relation_aware_retrieval and relation_ids is not None and role is not None
        fallback_count = 0
        relation_pool_size_entity_sum = 0.0
        source_entity_counts = {}

        if not use_relation_pool:
            # Default/global behavior is unchanged unless relation-aware retrieval is explicitly enabled with context.
            prototype_text_agg, retrieval_stat_tensors = self._knn_text_aggregation_from_pool(
                flat_entities,
                global_ids,
                text_emb.dtype,
            )
            source_entity_counts["global_text_pool"] = int(flat_entities.shape[0])
            topk = min(int(self.retrieval_topk), int(global_ids.numel()))
        else:
            prototype_text_agg = torch.zeros(flat_entities.shape[0], self.dim_e, device=device, dtype=text_emb.dtype)
            retrieval_stat_tensors = {
                "topk_similarity_mean": torch.zeros(flat_entities.shape[0], device=device, dtype=torch.float32),
                "topk_similarity_max": torch.zeros(flat_entities.shape[0], device=device, dtype=torch.float32),
                "topk_similarity_std": torch.zeros(flat_entities.shape[0], device=device, dtype=torch.float32),
            }
            topk = 0
            unique_relations = torch.unique(relation_ids).tolist()
            for relation_id in unique_relations:
                relation_mask = relation_ids == int(relation_id)
                relation_indices = torch.nonzero(relation_mask, as_tuple=False).view(-1)
                relation_entities = flat_entities.index_select(0, relation_indices)
                relation_pool = self._get_relation_text_pool(int(relation_id), role)
                relation_pool_size = int(relation_pool.numel()) if relation_pool is not None else 0
                relation_entity_count = int(relation_indices.numel())
                relation_pool_size_entity_sum += float(relation_pool_size * relation_entity_count)
                if relation_pool is None or relation_pool_size < self.min_relation_pool_size:
                    candidate_ids = global_ids
                    fallback_count += relation_entity_count
                    source = "fallback_global_text_pool"
                else:
                    candidate_ids = self._limit_retrieval_pool(relation_pool, device)
                    source = "relation_aware"
                source_entity_counts[source] = int(source_entity_counts.get(source, 0)) + relation_entity_count
                topk = max(topk, min(int(self.retrieval_topk), int(candidate_ids.numel())))
                relation_agg, relation_stats = self._knn_text_aggregation_from_pool(
                    relation_entities,
                    candidate_ids,
                    text_emb.dtype,
                )
                prototype_text_agg.index_copy_(0, relation_indices, relation_agg)
                for stat_name, stat_tensor in retrieval_stat_tensors.items():
                    stat_tensor.index_copy_(
                        0,
                        relation_indices,
                        relation_stats[stat_name].to(device=device, dtype=torch.float32),
                    )

        avg_relation_pool_size = (
            float(relation_pool_size_entity_sum / max(int(flat_entities.shape[0]), 1))
            if use_relation_pool else None
        )
        fallback_ratio = (
            float(fallback_count / max(int(flat_entities.shape[0]), 1))
            if use_relation_pool else 0.0
        )
        retrieval_source = "+".join(sorted(source_entity_counts.keys())) if source_entity_counts else self.retrieval_source
        retrieval_gate = self._build_confidence_retrieval_gate(retrieval_stat_tensors, text_emb)
        retrieval_confidence = self._build_retrieval_confidence(retrieval_stat_tensors, text_emb)
        self._update_retrieval_confidence_calibration_stats(
            retrieval_confidence,
            prototype_text_agg,
            replacement_mask,
        )
        gate_mean, missing_gate_mean, missing_text_count = self._summarize_retrieval_gate(
            retrieval_gate,
            replacement_mask,
        )

        def _stat_mean(stat_name):
            stat_tensor = retrieval_stat_tensors.get(stat_name)
            if stat_tensor is None or stat_tensor.numel() == 0:
                return None
            return float(stat_tensor.detach().float().mean().item())

        topk_similarity_mean_avg = _stat_mean("topk_similarity_mean")
        topk_similarity_max_avg = _stat_mean("topk_similarity_max")
        topk_similarity_std_avg = _stat_mean("topk_similarity_std")

        self._last_retrieval_missing_text_stats = {
            "phase": phase,
            "batch_entity_count": int(flat_entities.shape[0]),
            "topk": int(topk),
            "available_text_candidate_count": int(global_candidate_count),
            "retrieval_pool_size": int(global_ids.numel()),
            "use_relation_aware_retrieval": bool(use_relation_pool),
            "relation_role": role,
            "avg_relation_pool_size": avg_relation_pool_size,
            "fallback_ratio": fallback_ratio,
            "fallback_entity_count": int(fallback_count),
            "relation_pool_size_entity_sum": float(relation_pool_size_entity_sum),
            "retrieval_source_config": self.retrieval_source,
            "retrieval_source": retrieval_source,
            "source_entity_counts": source_entity_counts,
            "prototype_text_agg_mean_norm": float(prototype_text_agg.detach().norm(dim=-1).mean().item()),
            "use_confidence_gated_retrieval": bool(self.use_confidence_gated_retrieval),
            "retrieval_gate_type": self.retrieval_gate_type,
            "retrieval_gate_min": self.retrieval_gate_min,
            "retrieval_gate_max": self.retrieval_gate_max,
            "topk_similarity_mean_avg": topk_similarity_mean_avg,
            "topk_similarity_max_avg": topk_similarity_max_avg,
            "topk_similarity_std_avg": topk_similarity_std_avg,
            "retrieval_gate_mean": gate_mean,
            "missing_text_gate_mean": missing_gate_mean,
            "missing_text_entity_count": int(missing_text_count),
            "use_retrieval_confidence_calibration": bool(self.use_retrieval_confidence_calibration),
            "retrieval_confidence_type": self.retrieval_confidence_type,
            "retrieval_confidence_min_config": self.retrieval_confidence_min,
            "retrieval_confidence_max_config": self.retrieval_confidence_max,
            "retrieval_confidence_stats": self._last_retrieval_confidence_calibration_stats,
        }
        self._update_retrieval_missing_text_stats(self._last_retrieval_missing_text_stats)
        debug_key = "%s:%s" % (
            phase if phase is not None else "unknown",
            role if role is not None else "unknown",
        )
        if self._retrieval_missing_text_debug_counts_by_context.get(debug_key, 0) < 1:
            def _format_optional(value):
                return "NA" if value is None else "%.6f" % value

            gate_debug = ""
            if self.use_confidence_gated_retrieval:
                gate_debug = (
                    " | gate_mean=%s | missing_gate_mean=%s | topk_similarity_mean=%s"
                    % (
                        _format_optional(gate_mean),
                        _format_optional(missing_gate_mean),
                        _format_optional(topk_similarity_mean_avg),
                    )
                )
            print(
                "Retrieval missing-text debug | phase=%s | role=%s | batch_entity_count=%d | topk=%d | pool_size=%d/%d | relation_avg_pool=%s | fallback_ratio=%.4f | agg_mean_norm=%.6f | source=%s%s"
                % (
                    phase if phase is not None else "NA",
                    role if role is not None else "NA",
                    self._last_retrieval_missing_text_stats["batch_entity_count"],
                    self._last_retrieval_missing_text_stats["topk"],
                    self._last_retrieval_missing_text_stats["retrieval_pool_size"],
                    self._last_retrieval_missing_text_stats["available_text_candidate_count"],
                    (
                        "NA"
                        if avg_relation_pool_size is None else "%.2f" % avg_relation_pool_size
                    ),
                    fallback_ratio,
                    self._last_retrieval_missing_text_stats["prototype_text_agg_mean_norm"],
                    retrieval_source,
                    gate_debug,
                )
            )
            self._retrieval_missing_text_debug_counts_by_context[debug_key] = (
                self._retrieval_missing_text_debug_counts_by_context.get(debug_key, 0) + 1
            )
            self._retrieval_missing_text_debug_count += 1

        retrieval_gate = (
            retrieval_gate.to(device=device, dtype=text_emb.dtype).view(*text_emb.shape[:-1], 1)
            if retrieval_gate is not None else None
        )
        retrieval_confidence = (
            retrieval_confidence.to(device=device, dtype=text_emb.dtype).view(*text_emb.shape[:-1], 1)
            if retrieval_confidence is not None else None
        )
        return prototype_text_agg.to(device=device, dtype=text_emb.dtype).view_as(text_emb), retrieval_gate, retrieval_confidence

    def _get_branch_local_relation_gate(self, text_emb, context=None, replacement_mask=None):
        if not self.use_branch_local_relation_gate or not isinstance(context, dict):
            return None
        flat_count = text_emb.reshape(-1, text_emb.shape[-1]).shape[0]
        relation_ids = self._expand_relation_ids_for_retrieval(
            context.get("relation_ids"),
            flat_count,
            text_emb.device,
        )
        if relation_ids is None:
            return None

        if replacement_mask is None:
            missing_mask = torch.ones(flat_count, device=text_emb.device, dtype=torch.bool)
        else:
            missing_mask = replacement_mask.detach().view(-1).to(device=text_emb.device, dtype=torch.bool)
        missing_state = missing_mask.to(dtype=text_emb.dtype).unsqueeze(-1)
        relation_emb = self.rel_embeddings(relation_ids).to(device=text_emb.device, dtype=text_emb.dtype)
        # Branch-local relation gate: controls only missing-text compensation, not score routing.
        gate_input = torch.cat([relation_emb, missing_state], dim=-1)
        gate = torch.sigmoid(self.branch_local_relation_gate(gate_input)).view(*text_emb.shape[:-1], 1)

        gate_flat = gate.detach().view(-1).float()
        complete_mask = ~missing_mask
        missing_gate = gate_flat.masked_select(missing_mask)
        complete_gate = gate_flat.masked_select(complete_mask)
        self._last_branch_local_relation_gate_stats = {
            "missing_text_gate_mean": float(missing_gate.mean().item()) if missing_gate.numel() > 0 else None,
            "complete_text_gate_mean": float(complete_gate.mean().item()) if complete_gate.numel() > 0 else None,
            "gate_min": float(gate_flat.min().item()) if gate_flat.numel() > 0 else None,
            "gate_max": float(gate_flat.max().item()) if gate_flat.numel() > 0 else None,
            "missing_text_count": int(missing_mask.sum().item()),
            "complete_text_count": int(complete_mask.sum().item()),
        }
        if not self._branch_local_relation_gate_debug_printed:
            print(
                "Branch-local relation gate | missing_text_gate_mean=%s | complete_text_gate_mean=%s | gate_min=%s | gate_max=%s"
                % (
                    "NA" if self._last_branch_local_relation_gate_stats["missing_text_gate_mean"] is None else "%.6f" % self._last_branch_local_relation_gate_stats["missing_text_gate_mean"],
                    "NA" if self._last_branch_local_relation_gate_stats["complete_text_gate_mean"] is None else "%.6f" % self._last_branch_local_relation_gate_stats["complete_text_gate_mean"],
                    "NA" if self._last_branch_local_relation_gate_stats["gate_min"] is None else "%.6f" % self._last_branch_local_relation_gate_stats["gate_min"],
                    "NA" if self._last_branch_local_relation_gate_stats["gate_max"] is None else "%.6f" % self._last_branch_local_relation_gate_stats["gate_max"],
                )
            )
            self._branch_local_relation_gate_debug_printed = True
        return gate

    def _get_missing_text_replacements(self, text_emb, batch_entities, context=None, replacement_mask=None):
        if not self.use_prototype_missing_text:
            soft_missing_text = self._get_missing_text_token_value(text_emb, context=context).unsqueeze(0).expand_as(text_emb)
            if self.use_retrieval_missing_text:
                prototype_text_agg, retrieval_gate, retrieval_confidence = self._get_retrieval_missing_text_aggregation(
                    batch_entities,
                    text_emb,
                    context=context,
                    replacement_mask=replacement_mask,
                )
                if retrieval_confidence is not None:
                    retrieval_augmented_text = soft_missing_text + (
                        self.retrieval_mix_weight * retrieval_confidence * prototype_text_agg
                    )
                elif retrieval_gate is None:
                    retrieval_augmented_text = soft_missing_text + self.retrieval_mix_weight * prototype_text_agg
                else:
                    retrieval_augmented_text = soft_missing_text + retrieval_gate * prototype_text_agg
                retrieval_augmented_text = self._apply_cross_modal_text_imputer_residual(
                    batch_entities,
                    retrieval_augmented_text,
                    prototype_text_agg,
                    replacement_mask=replacement_mask,
                )
                branch_gate = self._get_branch_local_relation_gate(
                    text_emb,
                    context=context,
                    replacement_mask=replacement_mask,
                )
                if branch_gate is not None:
                    return (1.0 - branch_gate) * soft_missing_text + branch_gate * retrieval_augmented_text
                return retrieval_augmented_text
            return soft_missing_text

        if self.prototype_missing_text_token_bank is None or self.prototype_missing_text_cluster_ids is None:
            raise RuntimeError("Prototype missing-text is enabled but the token bank is not initialized.")

        entity_ids = batch_entities.detach().view(-1).cpu()
        cluster_ids = self.prototype_missing_text_cluster_ids.index_select(0, entity_ids)
        prototype_text = self.prototype_missing_text_token_bank(
            cluster_ids.to(device=text_emb.device, dtype=torch.long)
        )
        prototype_text = prototype_text.to(device=text_emb.device, dtype=text_emb.dtype)

        self._last_prototype_missing_text_stats = {
            "batch_entity_count": int(entity_ids.shape[0]),
            "unique_cluster_count": int(torch.unique(cluster_ids).numel()),
            "prototype_mean_norm": float(prototype_text.detach().norm(dim=-1).mean().item()),
        }
        if self._prototype_missing_text_debug_count < 3:
            print(
                "Prototype missing-text debug | batch_entity_count=%d | unique_cluster_count=%d | prototype_mean_norm=%.6f | context=soft_missing_path"
                % (
                    self._last_prototype_missing_text_stats["batch_entity_count"],
                    self._last_prototype_missing_text_stats["unique_cluster_count"],
                    self._last_prototype_missing_text_stats["prototype_mean_norm"],
                )
            )
            self._prototype_missing_text_debug_count += 1

        return prototype_text

    def _apply_soft_missing_text(self, text_emb, batch_entities, allow_pseudo_missing=True, context=None):
        if not self.uses_text_branch() or not self.use_soft_missing_text or self.has_text is None:
            return text_emb
        text_mask = self._get_entity_mask(batch_entities, self.has_text, text_emb.device)
        original_text_mask = self._get_entity_mask(batch_entities, self.original_has_text, text_emb.device)
        pseudo_missing_mask = None
        if (
            allow_pseudo_missing and
            self.training and
            self.pseudo_missing_prob > 0.0 and
            text_mask is not None and
            bool(text_mask.any().item())
        ):
            candidate_mask = text_mask.squeeze(-1)
            pseudo_missing_mask = candidate_mask & (
                torch.rand(candidate_mask.shape, device=text_emb.device) < self.pseudo_missing_prob
            )
            if bool(pseudo_missing_mask.any().item()):
                text_mask = text_mask & (~pseudo_missing_mask.unsqueeze(-1))
            if self._pseudo_missing_debug_count < 3:
                print(
                    "Pseudo-missing debug | prob=%.4f | eligible=%d | replaced=%d | mode=train"
                    % (
                        self.pseudo_missing_prob,
                        int(candidate_mask.sum().item()),
                        int(pseudo_missing_mask.sum().item()),
                    )
                )
                self._pseudo_missing_debug_count += 1
        oracle_restore_mask = self._get_oracle_restore_mask(batch_entities, text_emb.device)
        if oracle_restore_mask is not None:
            text_mask = text_mask | oracle_restore_mask
        if text_mask is None or bool(text_mask.all().item()):
            return text_emb
        if self.use_entity_specific_missing_text:
            surrogate_text = self._predict_entity_specific_missing_text(batch_entities)
            missing_count = int((~text_mask).sum().item())
            self._last_entity_specific_missing_text_stats = {
                "missing_count": missing_count,
                "surrogate_mean_norm": float(surrogate_text.norm(dim=-1).mean().item()),
            }
            if self._entity_specific_missing_text_debug_count < 3:
                print(
                    "Entity-specific missing-text debug | missing_count=%d | surrogate_mean_norm=%.6f | context=soft_missing_path"
                    % (
                        missing_count,
                        self._last_entity_specific_missing_text_stats["surrogate_mean_norm"],
                    )
                )
                self._entity_specific_missing_text_debug_count += 1
            return torch.where(text_mask, text_emb, surrogate_text)
        true_missing_mask = ~original_text_mask if original_text_mask is not None else ~text_mask
        proxy_mask = (~text_mask) & true_missing_mask
        shared_token_mask = (~text_mask) & (~proxy_mask)
        if self.use_structure_conditioned_missing_text:
            structure_proxy = self._predict_structure_conditioned_missing_text(batch_entities)
            self._last_structure_conditioned_missing_text_stats = {
                "missing_count": int((~text_mask).sum().item()),
                "proxy_count": int(proxy_mask.sum().item()),
                "shared_token_count": int(shared_token_mask.sum().item()),
            }
            if self._structure_conditioned_missing_text_debug_count < 3:
                print(
                    "Structure-conditioned missing-text debug | missing_count=%d | proxy_count=%d | shared_token_count=%d | context=soft_missing_path"
                    % (
                        self._last_structure_conditioned_missing_text_stats["missing_count"],
                        self._last_structure_conditioned_missing_text_stats["proxy_count"],
                        self._last_structure_conditioned_missing_text_stats["shared_token_count"],
                    )
                )
                self._structure_conditioned_missing_text_debug_count += 1
            output_text = torch.where(proxy_mask, structure_proxy, text_emb)
            if not bool(shared_token_mask.any().item()):
                return output_text
            missing_replacements = self._get_missing_text_replacements(
                text_emb,
                batch_entities,
                context=context,
                replacement_mask=shared_token_mask,
            )
            return torch.where(shared_token_mask, missing_replacements, output_text)
        missing_replacements = self._get_missing_text_replacements(
            text_emb,
            batch_entities,
            context=context,
            replacement_mask=~text_mask,
        )
        return torch.where(text_mask, text_emb, missing_replacements)

    def _get_text_reconstruction_mask(self, batch_entities, device):
        if self.has_text is None:
            return None
        text_mask = self._get_entity_mask(batch_entities, self.has_text, device)
        oracle_restore_mask = self._get_oracle_restore_mask(batch_entities, device)
        if oracle_restore_mask is not None:
            text_mask = text_mask | oracle_restore_mask
        return text_mask

    def _compute_entity_specific_missing_text_reconstruction_loss(self, batch_entities):
        if (
            not self.uses_text_branch() or
            not self.use_entity_specific_missing_text or
            self.entity_specific_missing_text_recon_weight <= 0.0
        ):
            return None
        unique_entities = torch.unique(batch_entities.detach().view(-1))
        if unique_entities.numel() == 0:
            return None
        target_text = self.text_proj(self.text_embeddings(unique_entities))
        target_text = self._apply_oracle_restore_text(target_text, unique_entities)
        supervision_mask = self._get_text_reconstruction_mask(unique_entities, target_text.device)
        if supervision_mask is None:
            return None
        supervision_mask = supervision_mask.squeeze(-1)
        if not bool(supervision_mask.any().item()):
            return None
        predicted_text = self._predict_entity_specific_missing_text(unique_entities)
        recon_loss = F.smooth_l1_loss(
            predicted_text[supervision_mask],
            target_text[supervision_mask],
        )
        return recon_loss * self.entity_specific_missing_text_recon_weight

    def consume_auxiliary_loss(self):
        aux_loss = self._pending_auxiliary_loss
        self._pending_auxiliary_loss = None
        return aux_loss

    def get_missing_text_gate(self):
        return torch.sigmoid(self.missing_text_gate_logit)

    def _get_entity_availability(self, batch_entities, entity_mask, device):
        if entity_mask is None:
            return torch.ones(batch_entities.view(-1).shape[0], 1, device=device, dtype=torch.float32)
        return self._get_entity_mask(batch_entities, entity_mask, device).to(dtype=torch.float32)

    def _get_pair_availability(self, batch_h, batch_t, entity_mask, device):
        h_availability = self._get_entity_availability(batch_h, entity_mask, device)
        t_availability = self._get_entity_availability(batch_t, entity_mask, device)
        return h_availability * t_availability

    def _get_flat_pair_availability_from_tensors(self, h_availability, t_availability, batch_r, mode):
        batch_size = batch_r.shape[0]
        h_availability = h_availability.view(-1)
        t_availability = t_availability.view(-1)

        if mode == "normal":
            if h_availability.shape[0] != t_availability.shape[0]:
                raise RuntimeError("Normal mode expects head and tail availability to have the same size.")
            return h_availability * t_availability

        if h_availability.shape[0] == batch_size and t_availability.shape[0] % batch_size == 0:
            neg_times = t_availability.shape[0] // batch_size
            h_availability = h_availability.view(1, batch_size).expand(neg_times, batch_size)
            t_availability = t_availability.view(neg_times, batch_size)
            return (h_availability * t_availability).reshape(-1)

        if t_availability.shape[0] == batch_size and h_availability.shape[0] % batch_size == 0:
            neg_times = h_availability.shape[0] // batch_size
            h_availability = h_availability.view(neg_times, batch_size)
            t_availability = t_availability.view(1, batch_size).expand(neg_times, batch_size)
            return (h_availability * t_availability).reshape(-1)

        raise RuntimeError(
            f"Unsupported pair-availability shapes: batch_h={h_availability.shape[0]}, batch_t={t_availability.shape[0]}, batch_r={batch_size}, mode={mode}"
        )

    def _get_flat_pair_availability(self, batch_h, batch_t, batch_r, mode, entity_mask, device):
        h_availability = self._get_entity_availability(batch_h, entity_mask, device).view(-1)
        t_availability = self._get_entity_availability(batch_t, entity_mask, device).view(-1)
        return self._get_flat_pair_availability_from_tensors(h_availability, t_availability, batch_r, mode)

    def _broadcast_availability_for_router(self, reference_tensor, availability, default_value=1.0):
        if availability is None:
            return torch.full_like(reference_tensor, default_value)
        availability = availability.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        while availability.dim() > reference_tensor.dim() and availability.shape[-1] == 1:
            availability = availability.squeeze(-1)
        while availability.dim() < reference_tensor.dim():
            availability = availability.unsqueeze(-1)
        return torch.ones_like(reference_tensor) * availability

    def _expand_query_side_availability(self, query_side_availability, candidate_side_availability, batch_r):
        if query_side_availability is None:
            return None
        query_side_availability = query_side_availability.view(-1)
        if candidate_side_availability is None:
            return query_side_availability

        batch_size = batch_r.shape[0]
        candidate_count = candidate_side_availability.view(-1).shape[0]
        if query_side_availability.shape[0] == candidate_count:
            return query_side_availability
        if query_side_availability.shape[0] != batch_size or candidate_count % batch_size != 0:
            raise RuntimeError(
                "Query-shared availability router received incompatible query/candidate shapes: "
                f"query={query_side_availability.shape[0]}, candidates={candidate_count}, batch={batch_size}"
            )
        candidate_times = candidate_count // batch_size
        return query_side_availability.view(1, batch_size).expand(candidate_times, batch_size).reshape(-1)

    def _get_query_score_availability(self, head_availability, tail_availability, batch_r, mode):
        # Query-shared router weights keep all candidates for the same ranking query comparable.
        # Tail prediction uses known-head availability; head prediction uses known-tail availability.
        if mode == "head_batch":
            return self._expand_query_side_availability(tail_availability, head_availability, batch_r)
        if mode == "tail_batch":
            return self._expand_query_side_availability(head_availability, tail_availability, batch_r)
        # Normal/score_all-style scoring has no single query side, so routing would make scores less comparable.
        return None

    def _get_availability_router_weights(self, reference_score, include_text_branch, image_availability=None, text_availability=None):
        if not self.use_availability_router:
            return None, None, None
        if self.availability_router_mode != "query_masked_softmax":
            raise ValueError(f"Unsupported availability_router_mode: {self.availability_router_mode}")
        if self.availability_router_eps <= 0.0:
            raise ValueError("availability_router_eps must be > 0.")

        # Availability-only router: structural is always available, image/text are masked by query availability.
        alpha_struct = torch.ones_like(reference_score)
        alpha_image = self._broadcast_availability_for_router(reference_score, image_availability, default_value=1.0)
        components = [alpha_struct, alpha_image]
        if include_text_branch:
            alpha_text = self._broadcast_availability_for_router(reference_score, text_availability, default_value=1.0)
            components.append(alpha_text)
        else:
            alpha_text = None

        # query_masked_softmax with zero logits is equivalent to masked renormalization over available modalities.
        router_inputs = torch.stack(components, dim=-1).clamp(min=0.0, max=1.0)
        router_denom = router_inputs.sum(dim=-1, keepdim=True).clamp_min(self.availability_router_eps)
        router_weights = router_inputs / router_denom
        alpha_struct = router_weights[..., 0]
        alpha_image = router_weights[..., 1]
        if include_text_branch:
            alpha_text = router_weights[..., 2]
        return alpha_struct, alpha_image, alpha_text

    def _combine_modality_scores_with_availability_router(
        self,
        score_s,
        score_i,
        score_t=None,
        image_availability=None,
        text_availability=None,
    ):
        include_text_branch = score_t is not None and self.uses_text_branch()
        alpha_struct, alpha_image, alpha_text = self._get_availability_router_weights(
            score_s,
            include_text_branch=include_text_branch,
            image_availability=image_availability,
            text_availability=text_availability,
        )
        if alpha_struct is None:
            return None

        routed_score = alpha_struct * score_s + alpha_image * score_i
        if include_text_branch:
            routed_score = routed_score + alpha_text * score_t
        return routed_score

    def _get_availability_router_active_mask(
        self,
        reference_score,
        image_availability=None,
        text_availability=None,
        include_text_branch=True,
    ):
        image_available = self._broadcast_availability_for_router(
            reference_score,
            image_availability,
            default_value=1.0,
        )
        active_mask = image_available < 0.5
        if include_text_branch:
            text_available = self._broadcast_availability_for_router(
                reference_score,
                text_availability,
                default_value=1.0,
            )
            active_mask = active_mask | (text_available < 0.5)
        return active_mask

    def _select_availability_routed_score(
        self,
        legacy_score,
        routed_score,
        image_availability=None,
        text_availability=None,
        include_text_branch=True,
    ):
        if routed_score is None:
            return legacy_score
        if image_availability is None and text_availability is None:
            return legacy_score
        # Preserve the learned fusion baseline for fully observed queries; only missing-query cases need routing.
        active_mask = self._get_availability_router_active_mask(
            legacy_score,
            image_availability=image_availability,
            text_availability=text_availability,
            include_text_branch=include_text_branch,
        )
        return torch.where(active_mask, routed_score, legacy_score)

    def _compute_legacy_joint_score_from_embeddings(
        self,
        h,
        t,
        r,
        mode,
        h_img_emb,
        t_img_emb,
        h_text_emb=None,
        t_text_emb=None,
        head_image_available=None,
        tail_image_available=None,
        head_text_available=None,
        tail_text_available=None,
    ):
        head_image_mask = None
        tail_image_mask = None
        head_text_mask = None
        tail_text_mask = None
        if self.use_availability_router:
            # Ranking uses one known query side and many candidates; only the known side gets availability routing.
            if mode == "tail_batch":
                head_image_mask = head_image_available.bool() if head_image_available is not None else None
                head_text_mask = head_text_available.bool() if h_text_emb is not None and head_text_available is not None else None
            elif mode == "head_batch":
                tail_image_mask = tail_image_available.bool() if tail_image_available is not None else None
                tail_text_mask = tail_text_available.bool() if t_text_emb is not None and tail_text_available is not None else None
        elif self._should_use_text_mask():
            head_text_mask = head_text_available.bool() if h_text_emb is not None and head_text_available is not None else None
            tail_text_mask = tail_text_available.bool() if t_text_emb is not None and tail_text_available is not None else None
        h_joint = self.get_joint_embeddings(
            h,
            h_img_emb,
            h_text_emb,
            text_mask=head_text_mask,
            image_mask=head_image_mask,
            text_available_mask=head_text_available,
            relation_embs=r,
        )
        t_joint = self.get_joint_embeddings(
            t,
            t_img_emb,
            t_text_emb,
            text_mask=tail_text_mask,
            image_mask=tail_image_mask,
            text_available_mask=tail_text_available,
            relation_embs=r,
        )
        return self.margin - self._calc(h_joint, t_joint, r, mode)

    def _combine_modality_scores(self, score_s, score_i, score_t=None, image_availability=None, text_availability=None):
        # Availability router is applied inside learned fusion attention, not on uncalibrated branch scores.
        if score_t is None or not self.uses_text_branch():
            if not self.use_missing_aware_joint_scoring and not self.use_masked_fixed_denominator_joint_scoring:
                return (score_s + score_i) / 2.0

            if image_availability is None:
                image_availability = torch.ones_like(score_s, dtype=score_s.dtype, device=score_s.device)
            image_availability = image_availability.to(device=score_s.device, dtype=score_s.dtype)
            if self.use_masked_fixed_denominator_joint_scoring:
                return (score_s + image_availability * score_i) / 2.0
            combined_score = score_s + image_availability * score_i
            normalizer = 1.0 + image_availability
            return combined_score / normalizer

        if not self.use_missing_aware_joint_scoring and not self.use_masked_fixed_denominator_joint_scoring:
            return (score_s + score_i + score_t) / 3.0

        if image_availability is None:
            image_availability = torch.ones_like(score_s, dtype=score_s.dtype, device=score_s.device)
        if text_availability is None:
            text_availability = torch.ones_like(score_s, dtype=score_s.dtype, device=score_s.device)

        image_availability = image_availability.to(device=score_s.device, dtype=score_s.dtype)
        text_availability = text_availability.to(device=score_s.device, dtype=score_s.dtype)
        if self.use_masked_fixed_denominator_joint_scoring:
            return (score_s + image_availability * score_i + text_availability * score_t) / 3.0
        combined_score = score_s + image_availability * score_i + text_availability * score_t
        normalizer = 1.0 + image_availability + text_availability
        return combined_score / normalizer

    def _score_stats(self, tensor):
        flat = tensor.detach().float().reshape(-1)
        return {
            "mean": float(flat.mean().item()),
            "var": float(flat.var(unbiased=False).item()),
        }

    def _availability_ratio(self, tensor):
        if tensor is None:
            return None
        flat = tensor.detach().float().reshape(-1)
        return float(flat.mean().item())

    def _denominator_stats(self, image_availability, text_availability):
        if image_availability is None:
            image_availability = 1.0
        if text_availability is None:
            text_availability = 0.0 if not self.uses_text_branch() else 1.0
        denominator = (1.0 + image_availability + text_availability).detach().float().reshape(-1)
        return {
            "active_modalities_mean": float(denominator.mean().item()),
            "active_modalities_var": float(denominator.var(unbiased=False).item()),
            "denominator_min": float(denominator.min().item()),
            "denominator_max": float(denominator.max().item()),
            "ratio_den_eq_1": float((denominator == 1.0).float().mean().item()),
            "ratio_den_eq_2": float((denominator == 2.0).float().mean().item()),
            "ratio_den_eq_3": float((denominator == 3.0).float().mean().item()),
        }

    def _maybe_debug_missing_aware_joint_scoring(
        self,
        phase,
        label,
        score_s,
        score_i,
        score_t,
        baseline_score,
        final_score,
        image_availability,
        text_availability,
        availability_summary=None,
    ):
        if not self.debug_missing_aware_joint_scoring or not self.use_missing_aware_joint_scoring:
            return

        counter_key = f"{phase}:{label}"
        current_count = self._missing_aware_joint_debug_counts.get(counter_key, 0)
        if current_count >= self.debug_missing_aware_joint_scoring_batches:
            return
        self._missing_aware_joint_debug_counts[counter_key] = current_count + 1

        score_s_stats = self._score_stats(score_s)
        score_i_stats = self._score_stats(score_i)
        score_t_stats = self._score_stats(score_t) if score_t is not None else None
        baseline_stats = self._score_stats(baseline_score)
        final_stats = self._score_stats(final_score)
        diff = (final_score.detach().float() - baseline_score.detach().float()).reshape(-1)
        denominator_stats = self._denominator_stats(image_availability, text_availability)

        print(
            f"Missing-aware joint scoring debug | phase={phase} | label={label} | step={current_count + 1}"
        )
        print(
            "  active_modalities | "
            f"mean={denominator_stats['active_modalities_mean']:.6f} | "
            f"var={denominator_stats['active_modalities_var']:.6f} | "
            f"min={denominator_stats['denominator_min']:.6f} | "
            f"max={denominator_stats['denominator_max']:.6f} | "
            f"ratio_den_eq_1={denominator_stats['ratio_den_eq_1']:.6f} | "
            f"ratio_den_eq_2={denominator_stats['ratio_den_eq_2']:.6f} | "
            f"ratio_den_eq_3={denominator_stats['ratio_den_eq_3']:.6f}"
        )
        if availability_summary:
            formatted = []
            for name, tensor in availability_summary.items():
                ratio = self._availability_ratio(tensor)
                if ratio is not None:
                    formatted.append(f"{name}={ratio:.6f}")
            if formatted:
                print("  availability_ratios | " + " | ".join(formatted))
        if score_t_stats is None:
            text_stats_summary = "text=skipped"
        else:
            text_stats_summary = f"text_mean={score_t_stats['mean']:.6f} text_var={score_t_stats['var']:.6f}"
        print(
            "  branch_score_stats | "
            f"struct_mean={score_s_stats['mean']:.6f} struct_var={score_s_stats['var']:.6f} | "
            f"image_mean={score_i_stats['mean']:.6f} image_var={score_i_stats['var']:.6f} | "
            f"{text_stats_summary}"
        )
        print(
            "  final_score_stats | "
            f"fixed_avg_mean={baseline_stats['mean']:.6f} fixed_avg_var={baseline_stats['var']:.6f} | "
            f"missing_aware_mean={final_stats['mean']:.6f} missing_aware_var={final_stats['var']:.6f} | "
            f"delta_mean={diff.mean().item():.6f} delta_var={diff.var(unbiased=False).item():.6f} | "
            f"delta_abs_mean={diff.abs().mean().item():.6f} delta_abs_max={diff.abs().max().item():.6f}"
        )

    def _broadcast_relation_expert_condition(self, relation_embs, target_emb):
        if relation_embs is None:
            return None
        relation_embs = relation_embs.to(device=target_emb.device, dtype=target_emb.dtype)
        if relation_embs.dim() == 2 and target_emb.dim() == 2 and relation_embs.shape[0] != target_emb.shape[0]:
            if target_emb.shape[0] % relation_embs.shape[0] != 0:
                return None
            relation_embs = relation_embs.repeat(target_emb.shape[0] // relation_embs.shape[0], 1)
        target_shape = list(target_emb.shape[:-1]) + [relation_embs.shape[-1]]
        while relation_embs.dim() < len(target_shape):
            relation_embs = relation_embs.unsqueeze(-2)
        return torch.broadcast_to(relation_embs, target_shape)

    def _broadcast_text_missing_state(self, text_available_mask, target_emb):
        target_shape = list(target_emb.shape[:-1]) + [1]
        if text_available_mask is None:
            return torch.zeros(target_shape, device=target_emb.device, dtype=target_emb.dtype)
        missing_state = 1.0 - text_available_mask.to(device=target_emb.device, dtype=target_emb.dtype)
        if missing_state.dim() == 2 and target_emb.dim() == 2 and missing_state.shape[0] != target_emb.shape[0]:
            if target_emb.shape[0] % missing_state.shape[0] != 0:
                return torch.zeros(target_shape, device=target_emb.device, dtype=target_emb.dtype)
            missing_state = missing_state.repeat(target_emb.shape[0] // missing_state.shape[0], 1)
        while missing_state.dim() < len(target_shape):
            missing_state = missing_state.unsqueeze(-1)
        return torch.broadcast_to(missing_state, target_shape)

    def _run_representation_expert(self, emb, expert_module):
        return emb + expert_module(emb)

    def _apply_missingness_relation_expert_fusion(
        self,
        es,
        ev,
        et=None,
        relation_embs=None,
        text_available_mask=None,
    ):
        # Representation-level expert fusion only; this is not a score router.
        if not self.use_missingness_relation_expert:
            return es, ev, et
        relation_condition = self._broadcast_relation_expert_condition(relation_embs, es)
        if relation_condition is None:
            return es, ev, et
        missing_state = self._broadcast_text_missing_state(text_available_mask, es)
        gate_input = torch.cat([relation_condition, missing_state], dim=-1)
        gate_logits = self.missingness_relation_expert_gate(
            gate_input.reshape(-1, gate_input.shape[-1])
        )
        gate_weights = torch.softmax(gate_logits, dim=-1).view(*gate_input.shape[:-1], self.expert_num)
        shared_weight = gate_weights[..., 0].unsqueeze(-1).to(dtype=es.dtype)
        missing_weight = gate_weights[..., 1].unsqueeze(-1).to(dtype=es.dtype)

        def _mix_modal_embedding(emb):
            if emb is None:
                return None
            shared_emb = self._run_representation_expert(emb, self.shared_representation_expert)
            missing_emb = self._run_representation_expert(emb, self.text_missing_representation_expert)
            return shared_weight * shared_emb + missing_weight * missing_emb

        es = _mix_modal_embedding(es)
        ev = _mix_modal_embedding(ev)
        et = _mix_modal_embedding(et)
        missing_mask = missing_state.squeeze(-1) > 0.5
        complete_mask = ~missing_mask

        def _masked_gate_mean(expert_index, mask):
            if not bool(mask.any().item()):
                return None
            return float(gate_weights[..., expert_index].detach()[mask].mean().item())

        self._last_missingness_relation_expert_stats = {
            "shared_expert_weight_mean": float(gate_weights[..., 0].detach().mean().item()),
            "text_missing_expert_weight_mean": float(gate_weights[..., 1].detach().mean().item()),
            "text_missing_state_mean": float(missing_state.detach().mean().item()),
            "complete_entity_count": int(complete_mask.sum().item()),
            "missing_text_entity_count": int(missing_mask.sum().item()),
            "complete_shared_expert_weight_mean": _masked_gate_mean(0, complete_mask),
            "complete_text_missing_expert_weight_mean": _masked_gate_mean(1, complete_mask),
            "missing_shared_expert_weight_mean": _masked_gate_mean(0, missing_mask),
            "missing_text_missing_expert_weight_mean": _masked_gate_mean(1, missing_mask),
            "relation_embedding_mean_norm": float(relation_condition.detach().norm(dim=-1).mean().item()),
        }
        if not self._missingness_relation_expert_debug_printed:
            print(
                "Missingness-relation expert fusion | representation_level=True | shared_weight_mean=%.6f | missing_weight_mean=%.6f | missing_state_mean=%.6f"
                % (
                    self._last_missingness_relation_expert_stats["shared_expert_weight_mean"],
                    self._last_missingness_relation_expert_stats["text_missing_expert_weight_mean"],
                    self._last_missingness_relation_expert_stats["text_missing_state_mean"],
                )
            )
            self._missingness_relation_expert_debug_printed = True
        return es, ev, et

    def get_attention(self, es, ev, et=None, text_mask=None, image_mask=None, text_available_mask=None):
        modality_embeddings = [es, ev]
        text_index = None
        if self.uses_text_branch() and et is not None:
            text_index = len(modality_embeddings)
            modality_embeddings.append(et)
        e = torch.stack(modality_embeddings, dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        if self.use_availability_router:
            modality_mask = torch.ones_like(scores, dtype=torch.bool)
            if image_mask is not None:
                modality_mask[:, 1] = image_mask.squeeze(-1)
            if text_index is not None and text_mask is not None:
                modality_mask[:, text_index] = text_mask.squeeze(-1)
            scores = scores.masked_fill(~modality_mask, -1e9)
        elif self.use_missing_mask:
            modality_mask = torch.ones_like(scores, dtype=torch.bool)
            if image_mask is not None:
                modality_mask[:, 1] = image_mask.squeeze(-1)
            if text_index is not None and text_mask is not None:
                modality_mask[:, text_index] = text_mask.squeeze(-1)
            scores = scores.masked_fill(~modality_mask, -1e9)
        baseline_attention = torch.softmax(scores, dim=-1)

        if self.use_availability_router:
            return baseline_attention

        if text_index is None or text_available_mask is None:
            return baseline_attention

        attention_weights = baseline_attention.clone()
        text_available = text_available_mask.to(device=attention_weights.device, dtype=attention_weights.dtype).squeeze(-1)

        if self.use_learnable_missing_text_gate:
            missing_scale = text_available + (1.0 - text_available) * self.get_missing_text_gate().to(attention_weights.dtype)
        elif self.use_missing_aware_fusion:
            missing_scale = text_available + (1.0 - text_available) * self.missing_text_attention_scale
        else:
            return baseline_attention

        modality_scale_components = [
            torch.ones_like(missing_scale),
            torch.ones_like(missing_scale),
        ]
        if text_index is not None:
            modality_scale_components.append(missing_scale)
        modality_scale = torch.stack(modality_scale_components, dim=-1)
        attention_weights = attention_weights * modality_scale
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        if self.debug_fusion_sanity and not self._fusion_debug_printed:
            attention_diff = (attention_weights - baseline_attention).abs().max(dim=-1).values
            broadcast_text_available = torch.broadcast_to(text_available, attention_diff.shape)
            full_mask = broadcast_text_available > 0.5
            missing_mask = ~full_mask

            def _format_diff(mask):
                if not bool(mask.any().item()):
                    return "none"
                masked_diff = attention_diff[mask]
                return f"max_abs_diff={masked_diff.max().item():.10f}, mean_abs_diff={masked_diff.mean().item():.10f}"

            print(
                "Fusion sanity | attention diff vs baseline | "
                f"full_text_samples: {_format_diff(full_mask)} | "
                f"missing_text_samples: {_format_diff(missing_mask)} | "
                f"missing_text_attention_scale={self.missing_text_attention_scale:.4f} | "
                f"learned_missing_text_gate={self.get_missing_text_gate().item():.6f}"
            )
            self._fusion_debug_printed = True

        return attention_weights

    def get_joint_embeddings(
        self,
        es,
        ev,
        et=None,
        text_mask=None,
        image_mask=None,
        text_available_mask=None,
        relation_embs=None,
    ):
        es, ev, et = self._apply_missingness_relation_expert_fusion(
            es,
            ev,
            et=et,
            relation_embs=relation_embs,
            text_available_mask=text_available_mask,
        )
        modality_embeddings = [es, ev]
        if self.uses_text_branch() and et is not None:
            modality_embeddings.append(et)
        e = torch.stack(modality_embeddings, dim=1)
        attention_weights = self.get_attention(
            es,
            ev,
            et,
            text_mask=text_mask,
            image_mask=image_mask,
            text_available_mask=text_available_mask
        )
        context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        return context_vectors
    

    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()
    
    def _calc_condition(self, h, t, r, mode):
        h = self.ent_embeddings(h)
        t = self.ent_embeddings(t)
        r = self.rel_embeddings(r)

        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_cond = re_relation * re_tail + im_relation * im_tail
            im_cond = re_relation * im_tail - im_relation * re_tail
        else:
            re_cond = re_head * re_relation - im_head * im_relation
            im_cond = re_head * im_relation + im_head * re_relation

        cond = torch.cat([re_cond, im_cond], dim=-1).squeeze(1)

        return cond

    def _compute_score_from_batch(
        self,
        batch_h,
        batch_t,
        batch_r,
        mode,
        phase,
        h_text_emb=None,
        t_text_emb=None,
        head_text_available=None,
        tail_text_available=None,
        allow_pseudo_missing=True,
        allow_debug=True,
    ):
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self._get_image_branch_embeddings(batch_h)
        t_img_emb = self._get_image_branch_embeddings(batch_t)
        if h_text_emb is None:
            h_text_emb = self._get_text_branch_embeddings(
                batch_h,
                self._make_text_context(f"{phase}_head", batch_r, "head", phase=("eval" if phase.startswith("eval") else "train")),
                allow_pseudo_missing=allow_pseudo_missing,
            )
        if t_text_emb is None:
            t_text_emb = self._get_text_branch_embeddings(
                batch_t,
                self._make_text_context(f"{phase}_tail", batch_r, "tail", phase=("eval" if phase.startswith("eval") else "train")),
                allow_pseudo_missing=allow_pseudo_missing,
            )
        head_image_available = self._get_entity_availability(batch_h, self.has_image, h.device)
        tail_image_available = self._get_entity_availability(batch_t, self.has_image, t.device)
        if head_text_available is None and self.uses_text_branch() and self.has_text is not None:
            head_text_available = self._get_entity_mask(batch_h, self.has_text, h.device)
        if tail_text_available is None and self.uses_text_branch() and self.has_text is not None:
            tail_text_available = self._get_entity_mask(batch_t, self.has_text, t.device)

        if self.use_missing_aware_joint_scoring or self.use_masked_fixed_denominator_joint_scoring:
            score_s = self.margin - self._calc(h, t, r, mode)
            score_i = self.margin - self._calc(h_img_emb, t_img_emb, r, mode)
            score_t = self.margin - self._calc(h_text_emb, t_text_emb, r, mode) if h_text_emb is not None and t_text_emb is not None else None
            if self.use_availability_router:
                image_availability = self._get_query_score_availability(
                    head_image_available,
                    tail_image_available,
                    batch_r,
                    mode,
                )
            else:
                image_availability = self._get_flat_pair_availability_from_tensors(
                    head_image_available.to(device=score_s.device, dtype=score_s.dtype),
                    tail_image_available.to(device=score_s.device, dtype=score_s.dtype),
                    batch_r,
                    mode,
                )
            text_availability = None
            if self.uses_text_branch() and head_text_available is not None and tail_text_available is not None:
                if self.use_availability_router:
                    text_availability = self._get_query_score_availability(
                        head_text_available,
                        tail_text_available,
                        batch_r,
                        mode,
                    )
                else:
                    text_availability = self._get_flat_pair_availability_from_tensors(
                        head_text_available.to(device=score_s.device, dtype=score_s.dtype),
                        tail_text_available.to(device=score_s.device, dtype=score_s.dtype),
                        batch_r,
                        mode,
                    )
            score = self._combine_modality_scores(score_s, score_i, score_t, image_availability, text_availability)
            if self.use_availability_router:
                legacy_score = self._compute_legacy_joint_score_from_embeddings(
                    h,
                    t,
                    r,
                    mode,
                    h_img_emb,
                    t_img_emb,
                    h_text_emb=h_text_emb,
                    t_text_emb=t_text_emb,
                    head_image_available=head_image_available,
                    tail_image_available=tail_image_available,
                    head_text_available=head_text_available,
                    tail_text_available=tail_text_available,
                )
                score = self._select_availability_routed_score(
                    legacy_score,
                    score,
                    image_availability=image_availability,
                    text_availability=text_availability,
                    include_text_branch=score_t is not None and self.uses_text_branch(),
                )
            if allow_debug and self.use_missing_aware_joint_scoring and not self.use_availability_router:
                baseline_score = (score_s + score_i + score_t) / 3.0 if score_t is not None else (score_s + score_i) / 2.0
                self._maybe_debug_missing_aware_joint_scoring(
                    phase=phase,
                    label="score",
                    score_s=score_s,
                    score_i=score_i,
                    score_t=score_t,
                    baseline_score=baseline_score,
                    final_score=score,
                    image_availability=image_availability,
                    text_availability=text_availability,
                    availability_summary={
                        "head_image_ratio": head_image_available,
                        "tail_image_ratio": tail_image_available,
                        "pair_image_ratio": image_availability,
                        "head_text_ratio": head_text_available,
                        "tail_text_ratio": tail_text_available,
                        "pair_text_ratio": text_availability,
                    },
                )
            return score

        return self._compute_legacy_joint_score_from_embeddings(
            h,
            t,
            r,
            mode,
            h_img_emb,
            t_img_emb,
            h_text_emb=h_text_emb,
            t_text_emb=t_text_emb,
            head_image_available=head_image_available,
            tail_image_available=tail_image_available,
            head_text_available=head_text_available,
            tail_text_available=tail_text_available,
        )

    def _compute_missing_text_consistency_loss(self, batch_h, batch_t, batch_r):
        if (
            not self.training or
            not self.uses_text_branch() or
            not self.use_soft_missing_text or
            not self.use_missing_text_consistency or
            self.consistency_lambda <= 0.0 or
            self.consistency_prob <= 0.0
        ):
            return None

        positive_count = batch_r.shape[0]
        if positive_count == 0:
            return None

        pos_batch_h = batch_h[:positive_count]
        pos_batch_t = batch_t[:positive_count]
        pos_batch_r = batch_r[:positive_count]
        device = pos_batch_r.device
        head_observed_mask = self._get_text_reconstruction_mask(pos_batch_h, device)
        tail_observed_mask = self._get_text_reconstruction_mask(pos_batch_t, device)
        if head_observed_mask is None or tail_observed_mask is None:
            return None

        head_observed_mask = head_observed_mask.squeeze(-1)
        tail_observed_mask = tail_observed_mask.squeeze(-1)
        candidate_pair_mask = head_observed_mask | tail_observed_mask
        if not bool(candidate_pair_mask.any().item()):
            return None

        sampled_pair_mask = candidate_pair_mask & (
            torch.rand(candidate_pair_mask.shape, device=device) < self.consistency_prob
        )
        if not bool(sampled_pair_mask.any().item()):
            return None

        both_observed_mask = sampled_pair_mask & head_observed_mask & tail_observed_mask
        head_only_mask = sampled_pair_mask & head_observed_mask & (~tail_observed_mask)
        tail_only_mask = sampled_pair_mask & tail_observed_mask & (~head_observed_mask)
        choose_head_for_both = torch.rand(both_observed_mask.shape, device=device) < 0.5
        replace_head_mask = head_only_mask | (both_observed_mask & choose_head_for_both)
        replace_tail_mask = tail_only_mask | (both_observed_mask & (~choose_head_for_both))
        if not bool(replace_head_mask.any().item() or replace_tail_mask.any().item()):
            return None

        sampled_indices = torch.nonzero(sampled_pair_mask, as_tuple=False).view(-1)
        sampled_batch_h = pos_batch_h.index_select(0, sampled_indices)
        sampled_batch_t = pos_batch_t.index_select(0, sampled_indices)
        sampled_batch_r = pos_batch_r.index_select(0, sampled_indices)
        sampled_replace_head = replace_head_mask.index_select(0, sampled_indices)
        sampled_replace_tail = replace_tail_mask.index_select(0, sampled_indices)

        observed_h_text = self._get_text_branch_embeddings(
            sampled_batch_h,
            self._make_text_context("train_consistency_observed_head", sampled_batch_r, "head", phase="train"),
            allow_pseudo_missing=False,
        )
        observed_t_text = self._get_text_branch_embeddings(
            sampled_batch_t,
            self._make_text_context("train_consistency_observed_tail", sampled_batch_r, "tail", phase="train"),
            allow_pseudo_missing=False,
        )
        if observed_h_text is None or observed_t_text is None:
            return None

        observed_score = self._compute_score_from_batch(
            sampled_batch_h,
            sampled_batch_t,
            sampled_batch_r,
            mode="normal",
            phase="train_consistency_observed",
            h_text_emb=observed_h_text,
            t_text_emb=observed_t_text,
            allow_pseudo_missing=False,
            allow_debug=False,
        )

        missing_token_h = self._get_missing_text_token_value(
            observed_h_text,
            context="train_consistency_missing_head",
        ).unsqueeze(0).expand_as(observed_h_text)
        missing_token_t = self._get_missing_text_token_value(
            observed_t_text,
            context="train_consistency_missing_tail",
        ).unsqueeze(0).expand_as(observed_t_text)
        missing_h_text = torch.where(sampled_replace_head.unsqueeze(-1), missing_token_h, observed_h_text)
        missing_t_text = torch.where(sampled_replace_tail.unsqueeze(-1), missing_token_t, observed_t_text)
        head_text_available = self._get_entity_mask(sampled_batch_h, self.has_text, observed_h_text.device)
        tail_text_available = self._get_entity_mask(sampled_batch_t, self.has_text, observed_t_text.device)
        if head_text_available is not None:
            head_text_available = head_text_available & (~sampled_replace_head.unsqueeze(-1))
        if tail_text_available is not None:
            tail_text_available = tail_text_available & (~sampled_replace_tail.unsqueeze(-1))

        missing_score = self._compute_score_from_batch(
            sampled_batch_h,
            sampled_batch_t,
            sampled_batch_r,
            mode="normal",
            phase="train_consistency_missing",
            h_text_emb=missing_h_text,
            t_text_emb=missing_t_text,
            head_text_available=head_text_available,
            tail_text_available=tail_text_available,
            allow_pseudo_missing=False,
            allow_debug=False,
        )

        consistency_loss = F.smooth_l1_loss(missing_score, observed_score.detach()) * self.consistency_lambda
        self._last_missing_text_consistency_stats = {
            "sampled_pair_count": int(sampled_pair_mask.sum().item()),
            "replace_head_count": int(replace_head_mask.sum().item()),
            "replace_tail_count": int(replace_tail_mask.sum().item()),
            "observed_score_mean": float(observed_score.detach().mean().item()),
            "missing_score_mean": float(missing_score.detach().mean().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
        }
        if self._missing_text_consistency_debug_count < 3:
            print(
                "Missing-text consistency debug | sampled_pairs=%d | replace_head=%d | replace_tail=%d | observed_mean=%.6f | missing_mean=%.6f | loss=%.6f"
                % (
                    self._last_missing_text_consistency_stats["sampled_pair_count"],
                    self._last_missing_text_consistency_stats["replace_head_count"],
                    self._last_missing_text_consistency_stats["replace_tail_count"],
                    self._last_missing_text_consistency_stats["observed_score_mean"],
                    self._last_missing_text_consistency_stats["missing_score_mean"],
                    self._last_missing_text_consistency_stats["consistency_loss"],
                )
            )
            self._missing_text_consistency_debug_count += 1
        return consistency_loss

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        self._pending_auxiliary_loss = None
        phase = "train" if "batch_y" in data else "eval"
        self._maybe_log_active_modalities(phase)
        score = self._compute_score_from_batch(
            batch_h,
            batch_t,
            batch_r,
            mode=mode,
            phase=f"{phase}_forward",
        )
        if "batch_y" in data:
            auxiliary_losses = []
            reconstruction_loss = self._compute_entity_specific_missing_text_reconstruction_loss(
                torch.cat([batch_h.view(-1), batch_t.view(-1)], dim=0)
            )
            if reconstruction_loss is not None:
                auxiliary_losses.append(reconstruction_loss)
            imputer_loss = self._compute_cross_modal_text_imputer_loss(
                torch.cat([batch_h.view(-1), batch_t.view(-1)], dim=0)
            )
            if imputer_loss is not None:
                auxiliary_losses.append(imputer_loss)
            consistency_loss = self._compute_missing_text_consistency_loss(batch_h, batch_t, batch_r)
            if consistency_loss is not None:
                auxiliary_losses.append(consistency_loss)
            if auxiliary_losses:
                total_auxiliary_loss = auxiliary_losses[0]
                for extra_loss in auxiliary_losses[1:]:
                    total_auxiliary_loss = total_auxiliary_loss + extra_loss
                self._pending_auxiliary_loss = total_auxiliary_loss
        return score
    
    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)
    
    def get_batch_rel_embs(self, data):
        return self.rel_embeddings(data)
    
    def get_batch_img_embs(self, data):
        return self._get_image_branch_embeddings(data)
    
    def get_batch_text_embs(self, data):
        if not self.uses_text_branch():
            self._maybe_log_text_branch_skipped("get_batch_text_embs")
            return None
        text_emb = self.text_proj(self.text_embeddings(data))
        text_emb = self._apply_oracle_restore_text(text_emb, data)
        text_emb = self._apply_fixed_zero_missing_text(text_emb, data)
        if not self.use_soft_token_text_generator_alignment:
            return text_emb
        text_emb = self._apply_soft_missing_text(text_emb, data)
        applied_soft_missing = self.use_soft_missing_text and self.has_text is not None
        self._maybe_log_generator_text_alignment(applied_soft_missing=applied_soft_missing)
        return text_emb

    def _condition_modal_embedding(self, emb, r, mode):
        pi = self.pi_const
        re_emb, im_emb = torch.chunk(emb, 2, dim=-1)
        phase_relation = r / (self.rel_embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "tail":
            re_cond = re_emb * re_relation - im_emb * im_relation
            im_cond = re_emb * im_relation + im_emb * re_relation
        else:
            re_cond = re_relation * re_emb + im_relation * im_emb
            im_cond = re_relation * im_emb - im_relation * re_emb

        return torch.cat([re_cond, im_cond], dim=-1)

    def get_modality_reliability(self, batch_entities, relation_embs, condition_mode):
        struct_emb = self.ent_embeddings(batch_entities)
        image_emb = self.img_proj(self.img_embeddings(batch_entities))

        image_available = self._get_entity_availability(batch_entities, self.has_image, struct_emb.device)
        text_available = self._get_entity_availability(batch_entities, self.has_text, struct_emb.device) if self.uses_text_branch() else torch.zeros_like(image_available)

        # Use a non-symmetric relation-aware compatibility:
        # compare relation-conditioned structural representation to raw modality representation.
        # This avoids the cosine invariance issue of applying the same RotatE transform to both sides.
        struct_cond = self._condition_modal_embedding(struct_emb, relation_embs, condition_mode)

        struct_norm = F.normalize(struct_cond, p=2, dim=-1, eps=1e-12)
        image_norm = F.normalize(image_emb, p=2, dim=-1, eps=1e-12)

        image_reliability = ((struct_norm * image_norm).sum(dim=-1, keepdim=True) + 1.0) / 2.0

        image_reliability = image_reliability * image_available
        if self.uses_text_branch():
            text_emb = self.text_proj(self.text_embeddings(batch_entities))
            text_emb = self._apply_oracle_restore_text(text_emb, batch_entities)
            text_norm = F.normalize(text_emb, p=2, dim=-1, eps=1e-12)
            text_reliability = ((struct_norm * text_norm).sum(dim=-1, keepdim=True) + 1.0) / 2.0
            text_reliability = text_reliability * text_available
        else:
            self._maybe_log_text_branch_skipped("get_modality_reliability")
            text_reliability = torch.zeros_like(image_reliability)

        return {
            "image_reliability": image_reliability,
            "text_reliability": text_reliability,
            "image_available": image_available,
            "text_available": text_available,
        }

    def _build_reliability_features(self, struct_cond, modal_emb):
        cosine = F.cosine_similarity(struct_cond, modal_emb, dim=-1, eps=1e-12).unsqueeze(-1)
        mean_abs_diff = (struct_cond - modal_emb).abs().mean(dim=-1, keepdim=True)
        struct_norm = struct_cond.norm(p=2, dim=-1, keepdim=True) / (self.dim_e ** 0.5)
        modal_norm = modal_emb.norm(p=2, dim=-1, keepdim=True) / (self.dim_e ** 0.5)
        return torch.cat([cosine, mean_abs_diff, struct_norm, modal_norm], dim=-1)

    def _get_entity_learned_reliability(self, batch_entities, relation_embs, condition_mode):
        struct_emb = self.ent_embeddings(batch_entities).detach()
        image_emb = self.img_proj(self.img_embeddings(batch_entities)).detach()

        relation_embs = relation_embs.detach()
        struct_cond = self._condition_modal_embedding(struct_emb, relation_embs, condition_mode).detach()

        image_features = self._build_reliability_features(struct_cond, image_emb)

        image_available = self._get_entity_availability(batch_entities, self.has_image, struct_emb.device)
        text_available = self._get_entity_availability(batch_entities, self.has_text, struct_emb.device) if self.uses_text_branch() else torch.zeros_like(image_available)

        image_reliability = torch.sigmoid(self.image_reliability_scorer(image_features)) * image_available
        if self.uses_text_branch():
            text_emb = self.text_proj(self.text_embeddings(batch_entities)).detach()
            text_emb = self._apply_oracle_restore_text(text_emb, batch_entities)
            text_features = self._build_reliability_features(struct_cond, text_emb)
            text_reliability = torch.sigmoid(self.text_reliability_scorer(text_features)) * text_available
        else:
            self._maybe_log_text_branch_skipped("_get_entity_learned_reliability")
            text_reliability = torch.zeros_like(image_reliability)
        return image_reliability, text_reliability

    def get_learned_reliability_conditioning(self, batch_h, batch_t, relation_embs):
        head_image_reliability, head_text_reliability = self._get_entity_learned_reliability(
            batch_h, relation_embs, condition_mode="tail"
        )
        tail_image_reliability, tail_text_reliability = self._get_entity_learned_reliability(
            batch_t, relation_embs, condition_mode="head"
        )
        if not self.uses_text_branch():
            return torch.cat(
                [
                    head_image_reliability,
                    tail_image_reliability,
                ],
                dim=-1,
            )
        return torch.cat(
            [
                head_image_reliability,
                head_text_reliability,
                tail_image_reliability,
                tail_text_reliability,
            ],
            dim=-1,
        )
    
    def get_neg_score(
        self,
        batch_h,
        batch_r, 
        batch_t,
        mode,
        fake_hv=None, 
        fake_tv=None,
        fake_ht=None,
        fake_tt=None
    ):
        if fake_hv is None or fake_tv is None:
            raise NotImplementedError
        if self.uses_text_branch() and (fake_ht is None or fake_tt is None):
            raise NotImplementedError
        self._maybe_log_active_modalities("train_negative")
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self._get_text_branch_embeddings(
            batch_h,
            self._make_text_context("get_neg_score_head", batch_r, "head", phase="train"),
        )
        t_text_emb = self._get_text_branch_embeddings(
            batch_t,
            self._make_text_context("get_neg_score_tail", batch_r, "tail", phase="train"),
        )
        head_text_available = self._get_entity_mask(batch_h, self.has_text, h.device) if self.uses_text_branch() and self.has_text is not None else None
        tail_text_available = self._get_entity_mask(batch_t, self.has_text, t.device) if self.uses_text_branch() and self.has_text is not None else None
        if self.use_missing_aware_joint_scoring or self.use_masked_fixed_denominator_joint_scoring:
            head_image_availability = self._get_entity_availability(batch_h, self.has_image, h.device)
            tail_image_availability = self._get_entity_availability(batch_t, self.has_image, t.device)
            pair_image_availability = head_image_availability * tail_image_availability
            pair_text_availability = head_text_available * tail_text_available if head_text_available is not None and tail_text_available is not None else None

            score_struct = self.margin - self._calc(h, t, r, mode)
            score_h_i = self.margin - self._calc(fake_hv, t_img_emb, r, mode)
            score_t_i = self.margin - self._calc(h_img_emb, fake_tv, r, mode)
            score_all_i = self.margin - self._calc(fake_hv, fake_tv, r, mode)
            score_h_t = self.margin - self._calc(fake_ht, t_text_emb, r, mode) if h_text_emb is not None and fake_ht is not None and t_text_emb is not None else None
            score_t_t = self.margin - self._calc(h_text_emb, fake_tt, r, mode) if h_text_emb is not None and fake_tt is not None and t_text_emb is not None else None
            score_all_t = self.margin - self._calc(fake_ht, fake_tt, r, mode) if fake_ht is not None and fake_tt is not None else None

            score_h = self._combine_modality_scores(score_struct, score_h_i, score_h_t, tail_image_availability, tail_text_available)
            score_t = self._combine_modality_scores(score_struct, score_t_i, score_t_t, head_image_availability, head_text_available)
            score_all = self._combine_modality_scores(
                score_struct,
                score_all_i,
                score_all_t,
                None if self.use_availability_router else pair_image_availability,
                None if self.use_availability_router else pair_text_availability,
            )
            if self.use_availability_router:
                h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, text_available_mask=head_text_available, relation_embs=r)
                t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, text_available_mask=tail_text_available, relation_embs=r)
                h_neg = self.get_joint_embeddings(h, fake_hv, fake_ht if self.uses_text_branch() else None, text_available_mask=head_text_available, relation_embs=r)
                t_neg = self.get_joint_embeddings(t, fake_tv, fake_tt if self.uses_text_branch() else None, text_available_mask=tail_text_available, relation_embs=r)
                legacy_score_h = self.margin - self._calc(h_neg, t_joint, r, mode)
                legacy_score_t = self.margin - self._calc(h_joint, t_neg, r, mode)
                legacy_score_all = self.margin - self._calc(h_neg, t_neg, r, mode)
                score_all = legacy_score_all
                score_h = self._select_availability_routed_score(
                    legacy_score_h,
                    score_h,
                    image_availability=tail_image_availability,
                    text_availability=tail_text_available,
                    include_text_branch=score_h_t is not None and self.uses_text_branch(),
                )
                score_t = self._select_availability_routed_score(
                    legacy_score_t,
                    score_t,
                    image_availability=head_image_availability,
                    text_availability=head_text_available,
                    include_text_branch=score_t_t is not None and self.uses_text_branch(),
                )
        else:
            h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, text_available_mask=head_text_available, relation_embs=r)
            t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, text_available_mask=tail_text_available, relation_embs=r)
            h_neg = self.get_joint_embeddings(h, fake_hv, fake_ht if self.uses_text_branch() else None, text_available_mask=head_text_available, relation_embs=r)
            t_neg = self.get_joint_embeddings(t, fake_tv, fake_tt if self.uses_text_branch() else None, text_available_mask=tail_text_available, relation_embs=r)
            score_h = self.margin - self._calc(h_neg, t_joint, r, mode)
            score_t = self.margin - self._calc(h_joint, t_neg, r, mode)
            score_all = self.margin - self._calc(h_neg, t_neg, r, mode)
        return [score_h, score_t, score_all], [h_img_emb, t_img_emb, h_text_emb, t_text_emb]
    
    def mm_negative_score(
        self,
        batch_h,
        batch_r, 
        batch_t,
        mode,
        w_margin,
        neg_h=None,
        neg_t=None,
        neg_hv=None, 
        neg_tv=None,
        neg_ht=None,
        neg_tt=None,
        batch_r_ids=None,
    ):
        if neg_hv is None or neg_tv is None:
            raise NotImplementedError
        if self.uses_text_branch() and (neg_ht is None or neg_tt is None):
            raise NotImplementedError
        self._maybe_log_active_modalities("train_negative")
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = batch_r
        h_img_emb = self._get_image_branch_embeddings(batch_h)
        t_img_emb = self._get_image_branch_embeddings(batch_t)
        h_text_emb = self._get_text_branch_embeddings(
            batch_h,
            self._make_text_context("mm_negative_score_head", batch_r_ids, "head", phase="train"),
        )
        t_text_emb = self._get_text_branch_embeddings(
            batch_t,
            self._make_text_context("mm_negative_score_tail", batch_r_ids, "tail", phase="train"),
        )
        head_text_available = self._get_entity_mask(batch_h, self.has_text, h.device) if self.uses_text_branch() and self.has_text is not None else None
        tail_text_available = self._get_entity_mask(batch_t, self.has_text, t.device) if self.uses_text_branch() and self.has_text is not None else None
        if self._should_use_text_mask() and self.has_text is not None:
            head_text_mask = self._get_entity_mask(batch_h, self.has_text, h.device)
            tail_text_mask = self._get_entity_mask(batch_t, self.has_text, t.device)
            pair_text_mask = head_text_mask & tail_text_mask
            self.last_mask_debug = {
                "head_missing_text_count": int((~head_text_mask).sum().item()),
                "tail_missing_text_count": int((~tail_text_mask).sum().item()),
                "score_all_text_masked_count": int((~pair_text_mask).sum().item()),
            }
        else:
            head_text_mask = None
            tail_text_mask = None
            self.last_mask_debug = None

        if self.use_missing_aware_joint_scoring or self.use_masked_fixed_denominator_joint_scoring:
            head_image_availability = self._get_entity_availability(batch_h, self.has_image, h.device)
            tail_image_availability = self._get_entity_availability(batch_t, self.has_image, t.device)
            pair_image_availability = head_image_availability * tail_image_availability
            head_text_availability = self._get_entity_availability(batch_h, self.has_text, h.device) if self.uses_text_branch() else None
            tail_text_availability = self._get_entity_availability(batch_t, self.has_text, t.device) if self.uses_text_branch() else None
            pair_text_availability = head_text_availability * tail_text_availability if self.uses_text_branch() else None

            h_pos = h.unsqueeze(1)
            t_pos = t.unsqueeze(1)
            h_img_pos = h_img_emb.unsqueeze(1)
            t_img_pos = t_img_emb.unsqueeze(1)
            h_text_pos = h_text_emb.unsqueeze(1) if h_text_emb is not None else None
            t_text_pos = t_text_emb.unsqueeze(1) if t_text_emb is not None else None

            score_h_s = w_margin * self.margin - self._calc(neg_h, t_pos, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
            score_h_i = w_margin * self.margin - self._calc(neg_hv, t_img_pos, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
            score_h_t = (
                w_margin * self.margin - self._calc(neg_ht, t_text_pos, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
                if h_text_pos is not None and neg_ht is not None else None
            )

            score_t_s = w_margin * self.margin - self._calc(h_pos, neg_t, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
            score_t_i = w_margin * self.margin - self._calc(h_img_pos, neg_tv, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
            score_t_t = (
                w_margin * self.margin - self._calc(h_text_pos, neg_tt, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
                if h_text_pos is not None and neg_tt is not None else None
            )

            score_all_s = w_margin * self.margin - self._calc(neg_h, neg_t, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
            score_all_i = w_margin * self.margin - self._calc(neg_hv, neg_tv, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
            score_all_t = (
                w_margin * self.margin - self._calc(neg_ht, neg_tt, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
                if neg_ht is not None and neg_tt is not None else None
            )

            score_h = self._combine_modality_scores(score_h_s, score_h_i, score_h_t, tail_image_availability, tail_text_availability)
            score_t = self._combine_modality_scores(score_t_s, score_t_i, score_t_t, head_image_availability, head_text_availability)
            score_all = self._combine_modality_scores(score_all_s, score_all_i, score_all_t, pair_image_availability, pair_text_availability)
            if self.use_availability_router:
                h_joint = self.get_joint_embeddings(
                    h,
                    h_img_emb,
                    h_text_emb,
                    text_mask=head_text_mask,
                    text_available_mask=head_text_availability,
                    relation_embs=r,
                ).unsqueeze(1)
                t_joint = self.get_joint_embeddings(
                    t,
                    t_img_emb,
                    t_text_emb,
                    text_mask=tail_text_mask,
                    text_available_mask=tail_text_availability,
                    relation_embs=r,
                ).unsqueeze(1)
                h_neg = self.get_joint_embeddings(
                    neg_h,
                    neg_hv,
                    neg_ht if self.uses_text_branch() else None,
                    text_mask=head_text_mask,
                    text_available_mask=head_text_availability,
                    relation_embs=r,
                )
                t_neg = self.get_joint_embeddings(
                    neg_t,
                    neg_tv,
                    neg_tt if self.uses_text_branch() else None,
                    text_mask=tail_text_mask,
                    text_available_mask=tail_text_availability,
                    relation_embs=r,
                )
                legacy_score_h = w_margin * self.margin - self._calc(h_neg, t_joint, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
                legacy_score_t = w_margin * self.margin - self._calc(h_joint, t_neg, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
                legacy_score_all = w_margin * self.margin - self._calc(h_neg, t_neg, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
                score_all = legacy_score_all
                score_h = self._select_availability_routed_score(
                    legacy_score_h,
                    score_h,
                    image_availability=tail_image_availability,
                    text_availability=tail_text_availability,
                    include_text_branch=score_h_t is not None and self.uses_text_branch(),
                )
                score_t = self._select_availability_routed_score(
                    legacy_score_t,
                    score_t,
                    image_availability=head_image_availability,
                    text_availability=head_text_availability,
                    include_text_branch=score_t_t is not None and self.uses_text_branch(),
                )
            if self.use_missing_aware_joint_scoring and not self.use_availability_router:
                baseline_score_h = (score_h_s + score_h_i + score_h_t) / 3.0 if score_h_t is not None else (score_h_s + score_h_i) / 2.0
                baseline_score_t = (score_t_s + score_t_i + score_t_t) / 3.0 if score_t_t is not None else (score_t_s + score_t_i) / 2.0
                baseline_score_all = (score_all_s + score_all_i + score_all_t) / 3.0 if score_all_t is not None else (score_all_s + score_all_i) / 2.0
                availability_summary = {
                    "head_image_ratio": head_image_availability,
                    "tail_image_ratio": tail_image_availability,
                    "pair_image_ratio": pair_image_availability,
                    "head_text_ratio": head_text_availability,
                    "tail_text_ratio": tail_text_availability,
                    "pair_text_ratio": pair_text_availability,
                }
                self._maybe_debug_missing_aware_joint_scoring(
                    phase="train_negative",
                    label="score_h",
                    score_s=score_h_s,
                    score_i=score_h_i,
                    score_t=score_h_t,
                    baseline_score=baseline_score_h,
                    final_score=score_h,
                    image_availability=head_image_availability,
                    text_availability=head_text_availability,
                    availability_summary=availability_summary,
                )
                self._maybe_debug_missing_aware_joint_scoring(
                    phase="train_negative",
                    label="score_t",
                    score_s=score_t_s,
                    score_i=score_t_i,
                    score_t=score_t_t,
                    baseline_score=baseline_score_t,
                    final_score=score_t,
                    image_availability=tail_image_availability,
                    text_availability=tail_text_availability,
                    availability_summary=availability_summary,
                )
                self._maybe_debug_missing_aware_joint_scoring(
                    phase="train_negative",
                    label="score_all",
                    score_s=score_all_s,
                    score_i=score_all_i,
                    score_t=score_all_t,
                    baseline_score=baseline_score_all,
                    final_score=score_all,
                    image_availability=pair_image_availability,
                    text_availability=pair_text_availability,
                    availability_summary=availability_summary,
                )
        else:
            h_joint = self.get_joint_embeddings(
                h,
                h_img_emb,
                h_text_emb,
                text_mask=head_text_mask,
                text_available_mask=head_text_available,
                relation_embs=r,
            ).unsqueeze(1)
            t_joint = self.get_joint_embeddings(
                t,
                t_img_emb,
                t_text_emb,
                text_mask=tail_text_mask,
                text_available_mask=tail_text_available,
                relation_embs=r,
            ).unsqueeze(1)
            h_neg = self.get_joint_embeddings(
                neg_h,
                neg_hv,
                neg_ht if self.uses_text_branch() else None,
                text_mask=head_text_mask,
                text_available_mask=head_text_available,
                relation_embs=r,
            )
            t_neg = self.get_joint_embeddings(
                neg_t,
                neg_tv,
                neg_tt if self.uses_text_branch() else None,
                text_mask=tail_text_mask,
                text_available_mask=tail_text_available,
                relation_embs=r,
            )
            score_h = w_margin * self.margin - self._calc(h_neg, t_joint, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
            score_t = w_margin * self.margin - self._calc(h_joint, t_neg, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
            score_all = w_margin * self.margin - self._calc(h_neg, t_neg, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
        return [score_h, score_t, score_all]

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def l3_regularization(self):
        return (self.ent_embeddings.weight.norm(p=3) ** 3 +
                self.rel_embeddings.weight.norm(p=3) ** 3)
    
    def get_attention_weight(self, h, t):
        device = next(self.parameters()).device
        h = torch.tensor([h], dtype=torch.long, device=device)
        t = torch.tensor([t], dtype=torch.long, device=device)
        self._maybe_log_active_modalities("attention_probe")
        h_s = self.ent_embeddings(h)
        t_s = self.ent_embeddings(t)
        h_img_emb = self.img_proj(self.img_embeddings(h))
        t_img_emb = self.img_proj(self.img_embeddings(t))
        h_text_emb = self._get_text_branch_embeddings(h, "get_attention_weight_head")
        t_text_emb = self._get_text_branch_embeddings(t, "get_attention_weight_tail")
        head_text_available = self._get_entity_mask(h, self.has_text, h_s.device) if self.uses_text_branch() and self.has_text is not None else None
        tail_text_available = self._get_entity_mask(t, self.has_text, t_s.device) if self.uses_text_branch() and self.has_text is not None else None
        head_text_mask = self._get_entity_mask(h, self.has_text, h_s.device) if h_text_emb is not None and self._should_use_text_mask() else None
        tail_text_mask = self._get_entity_mask(t, self.has_text, t_s.device) if t_text_emb is not None and self._should_use_text_mask() else None
        h_attn = self.get_attention(h_s, h_img_emb, h_text_emb, text_mask=head_text_mask, text_available_mask=head_text_available)
        t_attn = self.get_attention(t_s, t_img_emb, t_text_emb, text_mask=tail_text_mask, text_available_mask=tail_text_available)
        return h_attn, t_attn

    def export_fusion_probe(self, entity_ids, batch_size=4096):
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        entity_ids = torch.as_tensor(entity_ids, dtype=torch.long).view(-1).cpu()
        chunks = []

        with torch.no_grad():
            for start in range(0, int(entity_ids.numel()), batch_size):
                batch_cpu = entity_ids[start:start + batch_size]
                batch_entities = batch_cpu.to(device=device)
                structural_emb = self.ent_embeddings(batch_entities)
                image_emb = self._get_image_branch_embeddings(batch_entities)
                text_emb = self._get_text_branch_embeddings(
                    batch_entities,
                    "fusion_probe",
                    allow_pseudo_missing=False,
                )
                raw_text_emb = None
                if self.uses_text_branch():
                    raw_text_emb = self.text_proj(self.text_embeddings(batch_entities))
                    raw_text_emb = self._apply_oracle_restore_text(raw_text_emb, batch_entities)

                image_mask = None
                if self.use_availability_router and self.has_image is not None:
                    image_mask = self._get_entity_mask(batch_entities, self.has_image, structural_emb.device)
                text_available = (
                    self._get_entity_mask(batch_entities, self.has_text, structural_emb.device)
                    if self.uses_text_branch() and self.has_text is not None else None
                )
                text_mask = None
                if text_emb is not None and (self._should_use_text_mask() or self.use_availability_router):
                    text_mask = text_available.bool() if text_available is not None else None

                attention = self.get_attention(
                    structural_emb,
                    image_emb,
                    text_emb,
                    text_mask=text_mask,
                    image_mask=image_mask,
                    text_available_mask=text_available,
                )
                fused_emb = self.get_joint_embeddings(
                    structural_emb,
                    image_emb,
                    text_emb,
                    text_mask=text_mask,
                    image_mask=image_mask,
                    text_available_mask=text_available,
                )

                chunk = {
                    "entity_ids": batch_cpu,
                    "has_text": (
                        self.has_text.index_select(0, batch_cpu).cpu()
                        if self.has_text is not None else torch.ones_like(batch_cpu, dtype=torch.bool)
                    ),
                    "has_image": (
                        self.has_image.index_select(0, batch_cpu).cpu()
                        if self.has_image is not None else torch.ones_like(batch_cpu, dtype=torch.bool)
                    ),
                    "z_e": fused_emb.detach().cpu(),
                    "structural": structural_emb.detach().cpu(),
                    "visual": image_emb.detach().cpu(),
                    "attention": attention.detach().cpu(),
                }
                if text_emb is not None:
                    chunk["text"] = text_emb.detach().cpu()
                if raw_text_emb is not None:
                    chunk["raw_text"] = raw_text_emb.detach().cpu()
                chunks.append(chunk)

        if was_training:
            self.train()

        if not chunks:
            empty_long = torch.empty(0, dtype=torch.long)
            empty_bool = torch.empty(0, dtype=torch.bool)
            empty_float = torch.empty(0, self.dim_e)
            return {
                "entity_ids": empty_long,
                "has_text": empty_bool,
                "has_image": empty_bool,
                "z_e": empty_float,
                "structural": empty_float,
                "visual": empty_float,
                "attention": torch.empty(0, len(self.get_active_modalities())),
                "modalities": self.get_active_modalities(),
            }

        probe = {}
        for key in chunks[0]:
            probe[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
        probe["modalities"] = self.get_active_modalities()
        return probe

    def load_checkpoint(self, path):
        state_dict = torch.load(os.path.join(path))
        shared_token = state_dict.get("missing_text_token")
        if shared_token is not None:
            if "missing_text_token_head" not in state_dict:
                state_dict["missing_text_token_head"] = shared_token.clone()
            if "missing_text_token_tail" not in state_dict:
                state_dict["missing_text_token_tail"] = shared_token.clone()
        if self.use_missingness_relation_expert:
            expert_keys = [
                "missingness_relation_expert_gate.0.weight",
                "shared_representation_expert.0.weight",
                "text_missing_representation_expert.0.weight",
            ]
            if any(key not in state_dict for key in expert_keys):
                print(
                    "Checkpoint warning | missingness-relation expert params not found; "
                    "representation-level expert fusion will stay on fresh initialization."
                )
        self.load_state_dict(state_dict, strict=False)
        self.eval()
