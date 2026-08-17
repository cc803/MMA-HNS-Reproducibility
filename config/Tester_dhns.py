# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
from collections import defaultdict

class Tester_dhns(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True, other_model=None, norm=False, mu=0.5):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.other_model = other_model
        self.norm = norm
        self.mu = mu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def _read_triples(self, path):
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

    def _build_filtered_maps(self):
        in_path = self.data_loader.in_path
        all_triples = []
        for name in ["train2id.txt", "valid2id.txt", "test2id.txt"]:
            all_triples.extend(self._read_triples(os.path.join(in_path, name)))

        hr_to_tails = defaultdict(set)
        tr_to_heads = defaultdict(set)
        for h, t, r in all_triples:
            hr_to_tails[(h, r)].add(t)
            tr_to_heads[(t, r)].add(h)
        return hr_to_tails, tr_to_heads

    def _calc_filtered_rank(self, score, target_idx, filtered_ids):
        valid = np.ones(score.shape[0], dtype=bool)
        if filtered_ids:
            valid[list(filtered_ids)] = False
        valid[target_idx] = True
        target_score = score[target_idx]
        return int(np.sum(score[valid] < target_score)) + 1

    def _summarize_ranks(self, ranks):
        if len(ranks) == 0:
            return {
                "count": 0,
                "mrr": float("nan"),
                "mr": float("nan"),
                "hit10": float("nan"),
                "hit3": float("nan"),
                "hit1": float("nan"),
            }
        ranks = np.asarray(ranks, dtype=np.float64)
        return {
            "count": int(ranks.shape[0]),
            "mrr": float(np.mean(1.0 / ranks)),
            "mr": float(np.mean(ranks)),
            "hit10": float(np.mean(ranks <= 10)),
            "hit3": float(np.mean(ranks <= 3)),
            "hit1": float(np.mean(ranks <= 1)),
        }

    def _merge_group_metrics(self, metric_dict, group_names):
        merged_ranks = []
        merged_triple_count = 0
        for group_name in group_names:
            merged_ranks.extend(metric_dict[group_name]["_ranks"])
            merged_triple_count += metric_dict[group_name]["triple_count"]
        summary = self._summarize_ranks(merged_ranks)
        summary["count_type"] = "query"
        summary["triple_count"] = int(merged_triple_count)
        return summary


    def run_link_prediction(self, type_constrain=False, subset_eval=False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = self.data_loader
        subset_state = None
        has_text_tensor = getattr(self.model, "has_text", None)
        has_image_tensor = getattr(self.model, "has_image", None)
        if subset_eval and (has_text_tensor is not None or has_image_tensor is not None):
            has_text = has_text_tensor.cpu().numpy().astype(bool) if has_text_tensor is not None else None
            has_image = has_image_tensor.cpu().numpy().astype(bool) if has_image_tensor is not None else None
            injected_text_mask = None
            if getattr(self.model, "injected_text_mask", None) is not None:
                injected_text_mask = self.model.injected_text_mask.cpu().numpy().astype(bool)
            hr_to_tails, tr_to_heads = self._build_filtered_maps()
            groups = {}
            if has_text is not None:
                groups.update({
                    "head_missing_text": {"_ranks": [], "triple_count": 0},
                    "tail_missing_text": {"_ranks": [], "triple_count": 0},
                    "head_or_tail_missing_text": {"_ranks": [], "triple_count": 0},
                    "head_or_tail_injected_missing_text": {"_ranks": [], "triple_count": 0},
                    "head_and_tail_have_text": {"_ranks": [], "triple_count": 0},
                })
            if has_image is not None:
                groups.update({
                    "head_missing_image": {"_ranks": [], "triple_count": 0},
                    "tail_missing_image": {"_ranks": [], "triple_count": 0},
                    "head_or_tail_missing_image": {"_ranks": [], "triple_count": 0},
                    "head_and_tail_have_image": {"_ranks": [], "triple_count": 0},
                })
            subset_state = {
                "has_text": has_text,
                "has_image": has_image,
                "injected_text_mask": injected_text_mask,
                "hr_to_tails": hr_to_tails,
                "tr_to_heads": tr_to_heads,
                "overall_ranks": [],
                "groups": groups,
            }
        for index, [data_head, data_tail] in enumerate(training_range):
            head_score = self.test_one_step(data_head)
            self.lib.testHead(head_score.__array_interface__["data"][0], index, type_constrain)
            tail_score = self.test_one_step(data_tail)
            self.lib.testTail(tail_score.__array_interface__["data"][0], index, type_constrain)

            if subset_state is not None:
                h = int(data_tail['batch_h'][0])
                t = int(data_head['batch_t'][0])
                r = int(data_head['batch_r'][0])
                head_rank = self._calc_filtered_rank(head_score, h, subset_state["tr_to_heads"][(t, r)] - {h})
                tail_rank = self._calc_filtered_rank(tail_score, t, subset_state["hr_to_tails"][(h, r)] - {t})
                current_ranks = [head_rank, tail_rank]
                subset_state["overall_ranks"].extend(current_ranks)

                head_has_text = subset_state["has_text"][h] if subset_state["has_text"] is not None else None
                tail_has_text = subset_state["has_text"][t] if subset_state["has_text"] is not None else None
                head_has_image = subset_state["has_image"][h] if subset_state["has_image"] is not None else None
                tail_has_image = subset_state["has_image"][t] if subset_state["has_image"] is not None else None
                head_injected_missing = False
                tail_injected_missing = False
                if subset_state["injected_text_mask"] is not None:
                    head_injected_missing = subset_state["injected_text_mask"][h]
                    tail_injected_missing = subset_state["injected_text_mask"][t]
                if head_has_text is not None:
                    if not head_has_text:
                        subset_state["groups"]["head_missing_text"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["head_missing_text"]["triple_count"] += 1
                    if not tail_has_text:
                        subset_state["groups"]["tail_missing_text"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["tail_missing_text"]["triple_count"] += 1
                    if (not head_has_text) or (not tail_has_text):
                        subset_state["groups"]["head_or_tail_missing_text"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["head_or_tail_missing_text"]["triple_count"] += 1
                    if head_injected_missing or tail_injected_missing:
                        subset_state["groups"]["head_or_tail_injected_missing_text"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["head_or_tail_injected_missing_text"]["triple_count"] += 1
                    if head_has_text and tail_has_text:
                        subset_state["groups"]["head_and_tail_have_text"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["head_and_tail_have_text"]["triple_count"] += 1
                if head_has_image is not None:
                    if not head_has_image:
                        subset_state["groups"]["head_missing_image"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["head_missing_image"]["triple_count"] += 1
                    if not tail_has_image:
                        subset_state["groups"]["tail_missing_image"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["tail_missing_image"]["triple_count"] += 1
                    if (not head_has_image) or (not tail_has_image):
                        subset_state["groups"]["head_or_tail_missing_image"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["head_or_tail_missing_image"]["triple_count"] += 1
                    if head_has_image and tail_has_image:
                        subset_state["groups"]["head_and_tail_have_image"]["_ranks"].extend(current_ranks)
                        subset_state["groups"]["head_and_tail_have_image"]["triple_count"] += 1
        self.lib.test_link_prediction(type_constrain)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        overall_metrics = {
            "mrr": float(mrr),
            "mr": float(mr),
            "hit10": float(hit10),
            "hit3": float(hit3),
            "hit1": float(hit1),
        }
        if not subset_eval:
            return overall_metrics
        if subset_state is None:
            return overall_metrics, None, None

        subset_metrics = {}
        for group_name, group_state in subset_state["groups"].items():
            group_summary = self._summarize_ranks(group_state["_ranks"])
            group_summary["count_type"] = "query"
            group_summary["triple_count"] = int(group_state["triple_count"])
            subset_metrics[group_name] = group_summary
        if "head_and_tail_have_text" in subset_metrics:
            subset_metrics["head_or_tail_both_have_text"] = dict(subset_metrics["head_and_tail_have_text"])
        if "head_and_tail_have_image" in subset_metrics:
            subset_metrics["head_or_tail_both_have_image"] = dict(subset_metrics["head_and_tail_have_image"])

        query_level_overall = self._summarize_ranks(subset_state["overall_ranks"])
        query_level_overall["count_type"] = "query"
        query_level_overall["triple_count"] = int(len(subset_state["overall_ranks"]) // 2)

        sanity = {
            "count_meaning": "query_count counts head/tail prediction queries; triple_count counts original test triples.",
            "overall_query_metrics_same_loop": query_level_overall,
        }
        if "head_or_tail_missing_text" in subset_state["groups"] and "head_and_tail_have_text" in subset_state["groups"]:
            partition_overall = self._merge_group_metrics(
                subset_state["groups"],
                ["head_or_tail_missing_text", "head_and_tail_have_text"]
            )
            sanity["partition_recombined_metrics"] = partition_overall
            sanity["text_partition_recombined_metrics"] = partition_overall
        if "head_or_tail_missing_image" in subset_state["groups"] and "head_and_tail_have_image" in subset_state["groups"]:
            sanity["image_partition_recombined_metrics"] = self._merge_group_metrics(
                subset_state["groups"],
                ["head_or_tail_missing_image", "head_and_tail_have_image"]
            )
        return overall_metrics, subset_metrics, sanity

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod
