import collections
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from contextlib import nullcontext
import warnings

from mhcprime.model_utils import PeptideMHCDataset

class PepCL:
    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,

        steps: int = 200,
        output_dir: str = None,
        save_every_n_steps: int = 100,
        seed: int = 42,
        num_workers: int = 8,
        collate_fn = None,

        num_ms_pos: int = 50,
        ms_neg_ratio: float = 1.0,
        num_nonms_pos: int = 50,
        nonms_neg_ratio: float = 1.0,
        ms_batch_size: int = 3072,
        nonms_batch_size: int = 3072,

        auto_calibrate: bool = True,
        calibration_steps: int = 1, # for backward compat

        continuous_target_col: str = "ba_value",
        continuous_ft_sampling_mode: str = "allele_aware",  # "allele_aware" or "pooled"
        pooled_ft_num_pos: int = None,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        update_domain_embeddings: bool = False,

        use_replay: bool = True,
        primary_task_id: int = 1,
        sequential_backprop: bool = True,

        ms_loss_type: str = "logsmoothap",
        ms_loss_alpha: float = 10.0,
        ms_lambda: float = 1.0,
        nonms_loss_type: str = "logsmoothap",
        nonms_loss_alpha: float = 10.0,
        ft_lambda: float = 1.0,
        dr_lambda: float = 0.0,
        dr_loss_type: str = "mse",

        # Optional memory-efficient execution backend.
        # Default keeps the original PepCL implementation path unchanged.
        execution_mode: str = "legacy",  # "legacy" or "microbatch"
        use_microbatching: bool = False,
        ms_microbatch_size: int = 512,
        nonms_microbatch_size: int = 512,
        micro_strategy: str = "auto",
        batchwise_loss_types = None,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        preserve_cache_rng_state: bool = True,
        cache_forward_train_mode: bool = True,
        empty_cache_between_streams: bool = False,
        fail_on_nonfinite_loss: bool = True,
    ):
        self._pepcl_micro_engine = None
        _execution_mode = str(execution_mode).lower()
        if use_microbatching:
            _execution_mode = "microbatch"
        if _execution_mode in {"micro", "pepcl_micro", "pepcl-micro"}:
            _execution_mode = "microbatch"
        if _execution_mode not in {"legacy", "microbatch"}:
            raise ValueError("execution_mode must be one of: {'legacy', 'microbatch'}")

        if _execution_mode == "microbatch":
            self._pepcl_micro_engine = PepCL_Micro(
                model=model,
                train_df=train_df,
                tokenizer=tokenizer,
                device=device,
                steps=steps,
                output_dir=output_dir,
                save_every_n_steps=save_every_n_steps,
                seed=seed,
                num_workers=num_workers,
                collate_fn=collate_fn,
                num_ms_pos=num_ms_pos,
                ms_neg_ratio=ms_neg_ratio,
                num_nonms_pos=num_nonms_pos,
                nonms_neg_ratio=nonms_neg_ratio,
                ms_batch_size=ms_batch_size,
                nonms_batch_size=nonms_batch_size,
                ms_microbatch_size=ms_microbatch_size,
                nonms_microbatch_size=nonms_microbatch_size,
                auto_calibrate=auto_calibrate,
                calibration_steps=calibration_steps,
                continuous_target_col=continuous_target_col,
                continuous_ft_sampling_mode=continuous_ft_sampling_mode,
                pooled_ft_num_pos=pooled_ft_num_pos,
                encoder_lr=encoder_lr,
                decoder_lr=decoder_lr,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                update_domain_embeddings=update_domain_embeddings,
                use_replay=use_replay,
                primary_task_id=primary_task_id,
                sequential_backprop=sequential_backprop,
                ms_loss_type=ms_loss_type,
                ms_loss_alpha=ms_loss_alpha,
                ms_lambda=ms_lambda,
                nonms_loss_type=nonms_loss_type,
                nonms_loss_alpha=nonms_loss_alpha,
                ft_lambda=ft_lambda,
                dr_lambda=dr_lambda,
                dr_loss_type=dr_loss_type,
                micro_strategy=micro_strategy,
                batchwise_loss_types=batchwise_loss_types,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                preserve_cache_rng_state=preserve_cache_rng_state,
                cache_forward_train_mode=cache_forward_train_mode,
                empty_cache_between_streams=empty_cache_between_streams,
                fail_on_nonfinite_loss=fail_on_nonfinite_loss,
            )
            return

        self.model = model.to(device)
        self.df = train_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.device = device
        self.steps = steps
        self.output_dir = output_dir
        self.save_every_n_steps = save_every_n_steps
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        self.ms_loss_type = str(ms_loss_type).lower()
        self.nonms_loss_type = str(nonms_loss_type).lower()

        self.continuous_ft_loss_types = {"spearman", "mse"}
        self.ft_is_continuous = self.nonms_loss_type in self.continuous_ft_loss_types

        self.continuous_target_col = continuous_target_col
        self.continuous_ft_sampling_mode = continuous_ft_sampling_mode
        self.pooled_ft_num_pos = pooled_ft_num_pos

        if self.continuous_ft_sampling_mode not in {"allele_aware", "pooled"}:
            raise ValueError(
                "continuous_ft_sampling_mode must be one of: {'allele_aware', 'pooled'}"
            )

        if self.continuous_ft_sampling_mode == "pooled" and not self.ft_is_continuous:
            raise ValueError(
                "continuous_ft_sampling_mode='pooled' is only supported when "
                "nonms_loss_type is continuous, i.e. 'spearman' or 'mse'."
            )

        if self.ft_is_continuous and self.continuous_target_col not in self.df.columns:
            raise ValueError(
                f"Continuous FT requested because nonms_loss_type='{self.nonms_loss_type}', "
                f"but continuous_target_col='{self.continuous_target_col}' is not in train_df."
            )

        self.ms_bs, self.ft_bs = ms_batch_size, nonms_batch_size

        self.auto_calibrate = auto_calibrate
        self.calibration_steps = calibration_steps

        self.ms_cfg = {
            'n_pos': num_ms_pos,
            'ratio': ms_neg_ratio,
            'bs': self.ms_bs,
            'q_pos_by_allele': None,
            'sampling_summary': None,
            'sampling_mode': "allele_aware"
        }
        self.ft_cfg = {
            'n_pos': num_nonms_pos,
            'ratio': nonms_neg_ratio,
            'bs': self.ft_bs,
            'q_pos_by_allele': None,
            'sampling_summary': None,
            'sampling_mode': self.continuous_ft_sampling_mode if self.ft_is_continuous else "allele_aware",
            'pooled_ft_num_pos': self.pooled_ft_num_pos
        }

        if self.ft_is_continuous:
            self.base_dataset = PeptideMHCDataset(
                self.df,
                self.tokenizer,
                include_domain_id=True,
                feature_names=[self.continuous_target_col]
            )
        else:
            self.base_dataset = PeptideMHCDataset(
                self.df,
                self.tokenizer,
                include_domain_id=True
            )

        self.use_replay = use_replay
        self.primary_task_id = primary_task_id
        self.seq_backprop = sequential_backprop
        self.update_domain_embeddings = update_domain_embeddings
        self.dr_lambda = dr_lambda
        self.dr_loss_type = dr_loss_type
        self.ms_lambda, self.ft_lambda = ms_lambda, ft_lambda

        self.encoder_lr, self.decoder_lr = encoder_lr, decoder_lr
        self.weight_decay, self.warmup_ratio = weight_decay, warmup_ratio

        self.criterion_ms = self._get_criterion(self.ms_loss_type, ms_loss_alpha)
        self.criterion_ft = self._get_criterion(self.nonms_loss_type, nonms_loss_alpha)

        self.teacher = None
        if self.use_replay and self.dr_lambda > 0:
            self.teacher = copy.deepcopy(model).to(device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        torch.manual_seed(seed)
        np.random.seed(seed)

    class _LogSmoothAPScore(nn.Module):
        def __init__(self, alpha=10.0, epsilon=1e-8):
            super().__init__()
            self.alpha, self.epsilon = alpha, epsilon

        def forward(self, scores, labels):
            y = labels.float()
            P = y.sum().clamp(min=1.0)
            diff = scores.unsqueeze(0) - scores.unsqueeze(1)
            S = torch.sigmoid(self.alpha * diff)
            num = S.matmul(y)
            den = S.sum(dim=1).clamp(min=self.epsilon)
            soft_prec = num / den
            softap = (soft_prec * y).sum() / P
            return torch.log(1.0 / (softap + self.epsilon))

    @staticmethod
    def _soft_rank(x: torch.Tensor, tau: float = 1.0):
        """
        Differentiable rank based on pairwise sigmoid.
        r_i = 1 + Σ_j σ((x_j - x_i) / τ)
        τ -> 0 gives hard ranks.
        x : (..., L)
        returns (..., L) in [1, L]
        """
        x_col = x.unsqueeze(-1)
        x_row = x.unsqueeze(-2)
        P = torch.sigmoid((x_row - x_col) / tau)
        return 1 + P.sum(-1)

    class _SoftSpearmanLoss(nn.Module):
        def __init__(self, tau: float = 0.1):
            super().__init__()
            self.tau = tau

        def forward(self, pred: torch.Tensor, target: torch.Tensor):
            # ensure 2-D
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)

            r_pred = PepCL._soft_rank(pred, tau=self.tau)
            r_target = PepCL._soft_rank(target, tau=1e-4)

            r_pred = r_pred - r_pred.mean(-1, keepdim=True)
            r_target = r_target - r_target.mean(-1, keepdim=True)

            num = (r_pred * r_target).sum(-1)
            den = torch.norm(r_pred, dim=-1) * torch.norm(r_target, dim=-1) + 1e-8
            rho = num / den

            return (1 - rho).mean()

    class _TargetOverrideDataset(Dataset):
        """
        Wraps the parent PeptideMHCDataset so a stream can override:
            - domain_id
            - label target

        For classification streams:
            label remains the binary label from the parent dataset / dataframe.

        For continuous FT streams:
            label becomes peptide_{continuous_target_col}, e.g. peptide_ba_value,
            or falls back to df[continuous_target_col].
        """
        def __init__(
            self,
            parent_ds,
            df,
            domain_id,
            continuous_target: bool = False,
            continuous_target_col: str = "ba_value"
        ):
            self.parent_ds = parent_ds
            self.df = df
            self.domain_id = domain_id
            self.continuous_target = continuous_target
            self.continuous_target_col = continuous_target_col
            self.item_continuous_key = f"peptide_{continuous_target_col}"

        def __len__(self):
            return len(self.parent_ds)

        def __getitem__(self, idx):
            item = self.parent_ds[idx]
            item = dict(item)

            item['domain_id'] = torch.tensor(self.domain_id, dtype=torch.long)

            if self.continuous_target:
                if self.item_continuous_key in item:
                    target = item[self.item_continuous_key]
                    if isinstance(target, torch.Tensor):
                        item['label'] = target.float()
                    else:
                        item['label'] = torch.tensor(target, dtype=torch.float)
                else:
                    target = self.df.loc[idx, self.continuous_target_col]
                    item['label'] = torch.tensor(target, dtype=torch.float)

            return item

    class _InfiniteAlleleSampler(Sampler):
        """
        Infinite allele-aware sampler.

        Manual mode:
            Uses scalar n_pos per allele per sampler cycle.

        Auto mode:
            Uses q_pos_by_allele, a dict mapping allele -> positive quota per sampler cycle.

        Negative behavior:
            neg_ratio = 1.0 means sample equal number of negatives per selected positives.
            neg_ratio = 0.0 means sample no negatives.
            If an allele has no negatives, this is allowed and no negatives are sampled for that allele.
        """
        def __init__(
            self,
            df,
            domain_id,
            n_pos,
            neg_ratio,
            seed,
            q_pos_by_allele=None
        ):
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.n_pos = n_pos
            self.neg_ratio = float(neg_ratio)
            self.q_pos_by_allele = q_pos_by_allele

            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

            self.pools = {0: {}, 1: {}}

            subset = df[df['domain_id'] == domain_id]
            if subset.empty:
                self.alleles = []
                return

            for (lbl, allele), group in subset.groupby(['label', 'allele']):
                lbl_int = int(lbl)
                if lbl_int not in self.pools:
                    continue
                idx = group.index.to_numpy()
                self.rng.shuffle(idx)
                self.pools[lbl_int][allele] = idx

            positive_alleles = sorted(list(self.pools[1].keys()))

            if self.q_pos_by_allele is not None:
                self.q_pos_by_allele = {
                    al: int(q)
                    for al, q in self.q_pos_by_allele.items()
                    if int(q) > 0
                }
                self.alleles = [
                    al for al in positive_alleles
                    if self.q_pos_by_allele.get(al, 0) > 0
                ]
            else:
                self.alleles = positive_alleles

        def __iter__(self):
            if not self.alleles:
                return

            while True:
                indices = []

                for al in self.alleles:
                    pos_data = self.pools[1].get(al, [])
                    if len(pos_data) == 0:
                        continue

                    if self.q_pos_by_allele is not None:
                        requested_pos = int(self.q_pos_by_allele.get(al, 0))
                    else:
                        requested_pos = int(self.n_pos)

                    if requested_pos <= 0:
                        continue

                    p_take = min(len(pos_data), requested_pos)
                    if p_take <= 0:
                        continue

                    sel_pos = self.rng.choice(pos_data, size=p_take, replace=False)
                    indices.extend(sel_pos)

                    neg_data = self.pools[0].get(al, [])
                    n_req = int(len(sel_pos) * self.neg_ratio)

                    if n_req > 0 and len(neg_data) > 0:
                        n_take = min(len(neg_data), n_req)
                        if n_take > 0:
                            sel_neg = self.rng.choice(neg_data, size=n_take, replace=False)
                            indices.extend(sel_neg)

                if len(indices) == 0:
                    return

                self.rng.shuffle(indices)
                for i in indices:
                    yield i

    class _InfinitePooledFTSampler(Sampler):
        """
        Infinite global pooled sampler for continuous FT/BA mode.

        This mimics the important BA-side behavior of the older
        ContinuousRecyclingMTLDataLoaderFactory:

            - collect all domain 1 positives globally across alleles
            - collect all domain 1 negatives globally across alleles
            - sample pooled_ft_num_pos positives globally
            - sample negatives globally according to neg_ratio
            - yield shuffled indices forever

        The dataset wrapper handles using ba_value as the label target.
        """
        def __init__(
            self,
            df,
            domain_id,
            pooled_num_pos,
            neg_ratio,
            batch_size,
            seed
        ):
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.neg_ratio = float(neg_ratio)
            self.batch_size = int(batch_size)

            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

            subset = df[df['domain_id'] == domain_id].copy()
            if subset.empty:
                self.pos_pool = np.array([], dtype=int)
                self.neg_pool = np.array([], dtype=int)
                self.pooled_num_pos = 0
                return

            label_int = subset['label'].astype(int)

            self.pos_pool = subset.index[label_int == 1].to_numpy()
            self.neg_pool = subset.index[label_int == 0].to_numpy()

            if pooled_num_pos is None:
                # Default: one batch/cycle worth of positives implied by batch size and neg ratio.
                pooled_num_pos = int(self.batch_size / (1.0 + self.neg_ratio))
                pooled_num_pos = max(1, pooled_num_pos)

            self.pooled_num_pos = int(pooled_num_pos)

            self.rng.shuffle(self.pos_pool)
            self.rng.shuffle(self.neg_pool)

        def __iter__(self):
            if len(self.pos_pool) == 0 or self.pooled_num_pos <= 0:
                return

            while True:
                indices = []

                p_take = min(self.pooled_num_pos, len(self.pos_pool))
                if p_take > 0:
                    sel_pos = self.rng.choice(self.pos_pool, size=p_take, replace=False)
                    indices.extend(sel_pos)

                    n_req = int(len(sel_pos) * self.neg_ratio)

                    if n_req > 0 and len(self.neg_pool) > 0:
                        replace = len(self.neg_pool) < n_req
                        sel_neg = self.rng.choice(self.neg_pool, size=n_req, replace=replace)
                        indices.extend(sel_neg)

                if len(indices) == 0:
                    return

                self.rng.shuffle(indices)
                for i in indices:
                    yield i

    def _get_criterion(self, t, alpha):
        t = str(t).lower()

        if t == "logsmoothap":
            return self._LogSmoothAPScore(alpha=alpha)

        if t == "spearman":
            return self._SoftSpearmanLoss(tau=alpha)

        if t == "mse":
            return nn.MSELoss()

        if t in {"bce", "bcewithlogits", "bcewithlogitsloss"}:
            return nn.BCEWithLogitsLoss()

        return nn.BCEWithLogitsLoss()

    def _get_dr_loss(self, student_logits, teacher_logits):
        if self.dr_loss_type == "mse":
            return F.mse_loss(student_logits, teacher_logits)
        return F.mse_loss(student_logits, teacher_logits)

    def _get_optimizer(self):
        e, d = [], []
        for n, p in self.model.named_parameters():
            if "domain_embedder" in n and not self.update_domain_embeddings:
                p.requires_grad = False
                continue
            if not p.requires_grad:
                continue
            if any(x in n for x in ["embedder", "conformer", "transformer"]):
                e.append(p)
            else:
                d.append(p)

        return torch.optim.AdamW(
            [
                {'params': e, 'lr': self.encoder_lr},
                {'params': d, 'lr': self.decoder_lr}
            ],
            weight_decay=self.weight_decay
        )

    @staticmethod
    def _integer_quota_allocation(weights, caps, total, ensure_all_if_possible=True):
        """
        Allocate an integer total across alleles according to weights, respecting caps.
        """
        weights = np.asarray(weights, dtype=float)
        caps = np.asarray(caps, dtype=int)
        total = int(total)

        q = np.zeros_like(caps, dtype=int)

        valid = caps > 0
        if total <= 0 or not np.any(valid):
            return q

        max_total = int(min(total, caps.sum()))
        if max_total <= 0:
            return q

        w = np.zeros_like(weights, dtype=float)
        if weights[valid].sum() > 0:
            w[valid] = weights[valid] / weights[valid].sum()
        else:
            w[valid] = 1.0 / valid.sum()

        remaining_total = max_total

        if ensure_all_if_possible and remaining_total >= valid.sum():
            q[valid] = 1
            remaining_total -= int(valid.sum())

        if remaining_total <= 0:
            return q

        while remaining_total > 0:
            capacity = caps - q
            can_receive = capacity > 0

            if not np.any(can_receive):
                break

            w_active = np.zeros_like(w, dtype=float)
            if w[can_receive].sum() > 0:
                w_active[can_receive] = w[can_receive] / w[can_receive].sum()
            else:
                w_active[can_receive] = 1.0 / can_receive.sum()

            raw = remaining_total * w_active
            add = np.floor(raw).astype(int)
            add = np.minimum(add, capacity)

            if add.sum() > 0:
                q += add
                remaining_total -= int(add.sum())
                if remaining_total <= 0:
                    break

            capacity = caps - q
            can_receive = capacity > 0
            if not np.any(can_receive):
                break

            frac = raw - np.floor(raw)
            frac[~can_receive] = -1.0

            order = np.argsort(-frac)
            added_any = False

            for idx in order:
                if remaining_total <= 0:
                    break
                if capacity[idx] <= 0:
                    continue
                q[idx] += 1
                remaining_total -= 1
                added_any = True

            if not added_any:
                break

        return q

    def _compute_auto_domain_sampling(self, domain_id, batch_size, neg_ratio):
        """
        Data-adaptive allele-aware sampling calibration.

        This does not look at model loss and does not change gradient logic.
        It only computes per-allele positive quotas for batch construction.
        """
        if neg_ratio < 0:
            raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

        sub = self.df[self.df["domain_id"] == domain_id].copy()

        if sub.empty:
            return {}, pd.DataFrame()

        sub["_label_int_for_sampling"] = sub["label"].astype(int)

        counts = (
            sub.groupby(["allele", "_label_int_for_sampling"])
            .size()
            .unstack(fill_value=0)
        )

        n_pos = counts[1] if 1 in counts.columns else pd.Series(0, index=counts.index)
        n_neg = counts[0] if 0 in counts.columns else pd.Series(0, index=counts.index)

        summary = pd.DataFrame({
            "allele": counts.index,
            "n_pos": n_pos.astype(int).to_numpy(),
            "n_neg": n_neg.astype(int).to_numpy()
        })

        summary = summary[summary["n_pos"] > 0].reset_index(drop=True)

        if summary.empty:
            return {}, summary

        A = len(summary)
        P = summary["n_pos"].astype(float).to_numpy()
        caps = summary["n_pos"].astype(int).to_numpy()

        target_pos = int(batch_size / (1.0 + float(neg_ratio)))
        target_pos = max(1, target_pos)

        uniform = np.ones(A, dtype=float) / A
        prevalence = P / P.sum()

        denom = np.square(P).sum()
        if denom <= 0:
            A_eff = float(A)
        else:
            A_eff = float((P.sum() ** 2) / denom)

        balance_score = float(A_eff / A)
        balance_score = max(0.0, min(1.0, balance_score))

        weights = balance_score * uniform + (1.0 - balance_score) * prevalence
        weights = weights / weights.sum()

        q_pos = self._integer_quota_allocation(
            weights=weights,
            caps=caps,
            total=target_pos,
            ensure_all_if_possible=False
        )

        q_pos_by_allele = {
            al: int(q)
            for al, q in zip(summary["allele"].tolist(), q_pos)
            if int(q) > 0
        }

        summary["allele_weight"] = weights
        summary["q_pos"] = q_pos.astype(int)
        summary["expected_q_neg_if_available"] = np.floor(
            summary["q_pos"].to_numpy() * float(neg_ratio)
        ).astype(int)
        summary["reuse_pressure_pos"] = summary["q_pos"] / summary["n_pos"].clip(lower=1)

        summary["has_negatives"] = summary["n_neg"] > 0

        summary.attrs["target_pos"] = target_pos
        summary.attrs["n_alleles"] = A
        summary.attrs["effective_n_alleles"] = A_eff
        summary.attrs["balance_score"] = balance_score

        return q_pos_by_allele, summary

    def _compute_pooled_ft_summary(self):
        """
        Diagnostic summary for pooled continuous FT mode.
        This does not affect sampling except determining the default pooled_ft_num_pos if None.
        """
        sub = self.df[self.df["domain_id"] == 1].copy()

        if sub.empty:
            return pd.DataFrame()

        sub["_label_int_for_sampling"] = sub["label"].astype(int)

        n_pos_total = int((sub["_label_int_for_sampling"] == 1).sum())
        n_neg_total = int((sub["_label_int_for_sampling"] == 0).sum())
        n_alleles_pos = int(sub.loc[sub["_label_int_for_sampling"] == 1, "allele"].nunique())
        n_alleles_all = int(sub["allele"].nunique())

        if self.pooled_ft_num_pos is None:
            pooled_pos = int(self.ft_cfg["bs"] / (1.0 + float(self.ft_cfg["ratio"])))
            pooled_pos = max(1, pooled_pos)
        else:
            pooled_pos = int(self.pooled_ft_num_pos)

        pooled_neg = int(pooled_pos * float(self.ft_cfg["ratio"]))

        summary = pd.DataFrame({
            "mode": ["pooled"],
            "n_pos_total": [n_pos_total],
            "n_neg_total": [n_neg_total],
            "n_alleles_pos": [n_alleles_pos],
            "n_alleles_all": [n_alleles_all],
            "pooled_ft_num_pos": [pooled_pos],
            "expected_pooled_num_neg": [pooled_neg],
            "pos_reuse_pressure": [pooled_pos / max(n_pos_total, 1)]
        })

        summary.attrs["pooled_ft_num_pos"] = pooled_pos
        summary.attrs["expected_pooled_num_neg"] = pooled_neg

        return summary

    def _auto_calibrate(self):
        """
        Auto-calibration only controls sampling parameters.

        For MS:
            always allele-aware.

        For FT:
            - allele_aware: compute per-allele positive quotas.
            - pooled: do not compute per-allele quotas; print pooled diagnostics.
        """
        ms_q, ms_summary = self._compute_auto_domain_sampling(
            domain_id=0,
            batch_size=self.ms_cfg["bs"],
            neg_ratio=self.ms_cfg["ratio"]
        )

        self.ms_cfg["q_pos_by_allele"] = ms_q
        self.ms_cfg["sampling_summary"] = ms_summary

        if self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
            ft_summary = self._compute_pooled_ft_summary()
            self.ft_cfg["q_pos_by_allele"] = None
            self.ft_cfg["sampling_summary"] = ft_summary
        else:
            ft_q, ft_summary = self._compute_auto_domain_sampling(
                domain_id=1,
                batch_size=self.ft_cfg["bs"],
                neg_ratio=self.ft_cfg["ratio"]
            )
            self.ft_cfg["q_pos_by_allele"] = ft_q
            self.ft_cfg["sampling_summary"] = ft_summary

        def _summary_line(name, summary, cfg):
            if summary is None or summary.empty:
                return f"  {name}: no eligible positive alleles"

            if "mode" in summary.columns and summary["mode"].iloc[0] == "pooled":
                row = summary.iloc[0]
                return (
                    f"  {name}: pooled continuous FT, "
                    f"alleles_pos={int(row['n_alleles_pos'])}, alleles_all={int(row['n_alleles_all'])}, "
                    f"pos_pool={int(row['n_pos_total'])}, neg_pool={int(row['n_neg_total'])}, "
                    f"pooled_pos/cycle={int(row['pooled_ft_num_pos'])}, "
                    f"expected_neg/cycle={int(row['expected_pooled_num_neg'])}, "
                    f"pos_reuse={float(row['pos_reuse_pressure']):.4f}, "
                    f"neg_ratio={cfg['ratio']}"
                )

            # Allele-aware summary
            target_pos = summary.attrs.get("target_pos", None)
            A = summary.attrs.get("n_alleles", len(summary))
            A_eff = summary.attrs.get("effective_n_alleles", np.nan)
            balance_score = summary.attrs.get("balance_score", np.nan)

            q_nonzero = summary.loc[summary["q_pos"] > 0, "q_pos"]
            reuse = summary.loc[summary["q_pos"] > 0, "reuse_pressure_pos"]

            q_med = float(q_nonzero.median()) if len(q_nonzero) else 0.0
            q_min = int(q_nonzero.min()) if len(q_nonzero) else 0
            q_max = int(q_nonzero.max()) if len(q_nonzero) else 0

            reuse_med = float(reuse.median()) if len(reuse) else 0.0
            reuse_max = float(reuse.max()) if len(reuse) else 0.0

            return (
                f"  {name}: alleles={A}, A_eff={A_eff:.2f}, balance={balance_score:.3f}, "
                f"target_pos/cycle={target_pos}, q_pos median={q_med:.1f} "
                f"[{q_min}, {q_max}], reuse median={reuse_med:.4f}, reuse max={reuse_max:.4f}, "
                f"neg_ratio={cfg['ratio']}"
            )

        print("[PepCL] Auto-calibrated sampling:")
        print(_summary_line("MS", self.ms_cfg["sampling_summary"], self.ms_cfg))
        print(_summary_line("FT", self.ft_cfg["sampling_summary"], self.ft_cfg))
        print("")

    def train(self):
        if getattr(self, "_pepcl_micro_engine", None) is not None:
            return self._pepcl_micro_engine.train()

        if self.auto_calibrate:
            self._auto_calibrate()
        else:
            if self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
                self.ft_cfg["sampling_summary"] = self._compute_pooled_ft_summary()

            print(
                f"[PepCL] Manual mode: "
                f"MS n_pos={self.ms_cfg['n_pos']}, MS neg_ratio={self.ms_cfg['ratio']} | "
                f"FT n_pos={self.ft_cfg['n_pos']}, FT neg_ratio={self.ft_cfg['ratio']} | "
                f"FT continuous={self.ft_is_continuous}, "
                f"FT sampling_mode={self.ft_cfg['sampling_mode']}\n"
            )

        def _build_loader(domain_id, cfg, seed):
            continuous_target = (
                domain_id == 1 and
                self.ft_is_continuous
            )

            ds = self._TargetOverrideDataset(
                parent_ds=self.base_dataset,
                df=self.df,
                domain_id=domain_id,
                continuous_target=continuous_target,
                continuous_target_col=self.continuous_target_col
            )

            if (
                domain_id == 1 and
                self.ft_is_continuous and
                cfg.get("sampling_mode", "allele_aware") == "pooled"
            ):
                sampler = self._InfinitePooledFTSampler(
                    self.df,
                    domain_id=domain_id,
                    pooled_num_pos=cfg.get("pooled_ft_num_pos", None),
                    neg_ratio=cfg["ratio"],
                    batch_size=cfg["bs"],
                    seed=seed
                )
            else:
                sampler = self._InfiniteAlleleSampler(
                    self.df,
                    domain_id,
                    cfg['n_pos'],
                    cfg['ratio'],
                    seed,
                    q_pos_by_allele=cfg.get("q_pos_by_allele", None)
                )

            return DataLoader(
                ds,
                batch_size=cfg['bs'],
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                persistent_workers=(self.num_workers > 0)
            )

        ms_iter = iter(_build_loader(0, self.ms_cfg, self.seed))
        ft_iter = iter(_build_loader(1, self.ft_cfg, self.seed + 1))

        optimizer = self._get_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(self.steps * self.warmup_ratio),
            self.steps
        )
        scaler = GradScaler()

        metrics = collections.defaultdict(list)
        pbar = tqdm(range(self.steps), desc="PepCL")

        for step in pbar:
            self.model.train()
            optimizer.zero_grad()

            l_ms_v, l_ft_v, l_dr_v = 0.0, 0.0, 0.0

            # Safety initialization for non-sequential mode.
            # This prevents stale tensors if one stream is unavailable in a given step.
            loss_ms = None
            loss_ft = None

            # Domain 0 / Replay
            if self.use_replay or self.primary_task_id == 0:
                try:
                    batch_ms = next(ms_iter)
                    batch_ms = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_ms.items()
                    }
                    lbl_ms = batch_ms.pop('label')

                    with autocast(self.device):
                        logits_ms = self.model(batch_ms).view(-1)
                        loss_ms = self.criterion_ms(logits_ms, lbl_ms) * self.ms_lambda

                        if self.dr_lambda > 0 and self.teacher is not None:
                            with torch.no_grad():
                                t_logits = self.teacher(batch_ms).view(-1)
                            loss_dr = self._get_dr_loss(logits_ms, t_logits) * self.dr_lambda
                            loss_ms += loss_dr
                            l_dr_v = loss_dr.item()

                    if self.seq_backprop:
                        scaler.scale(loss_ms).backward()

                    l_ms_v = loss_ms.item()

                except (StopIteration, AttributeError):
                    pass

            if self.use_replay or self.primary_task_id == 1:
                try:
                    batch_ft = next(ft_iter)
                    batch_ft = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_ft.items()
                    }
                    lbl_ft = batch_ft.pop('label')

                    with autocast(self.device):
                        logits_ft = self.model(batch_ft).view(-1)
                        loss_ft = self.criterion_ft(logits_ft, lbl_ft) * self.ft_lambda

                    if self.seq_backprop:
                        scaler.scale(loss_ft).backward()

                    l_ft_v = loss_ft.item()

                except (StopIteration, AttributeError):
                    pass

            if not self.seq_backprop:
                total_l = None

                if loss_ms is not None:
                    total_l = loss_ms

                if loss_ft is not None:
                    total_l = loss_ft if total_l is None else total_l + loss_ft

                if total_l is not None:
                    scaler.scale(total_l).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix({
                'MS': f"{l_ms_v:.6f}",
                'FT': f"{l_ft_v:.6f}",
                'DR': f"{l_dr_v:.6f}"
            })

            metrics['ms_loss'].append(l_ms_v)
            metrics['ft_loss'].append(l_ft_v)
            metrics['dr_loss'].append(l_dr_v)

            if self.output_dir and (step + 1) % self.save_every_n_steps == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, f"pepcl_step_{step+1}.pt")
                )

        return metrics


# -----------------------------------------------------------------------------
# Memory-efficient PepCL backend.
# This class is intentionally kept separate from the legacy PepCL training loop.
# PepCL(execution_mode="microbatch") routes here, while execution_mode="legacy"
# executes the original publication implementation above.
# -----------------------------------------------------------------------------

class PepCL_Micro:
    """
    Memory-efficient PepCL variant for smaller GPUs.

    Main idea
    ---------
    PepCL_Micro preserves the *logical* stream batch size used by PepCL
    (default 3072) while using smaller *physical* model microbatches
    (for example 256/512/768/1024) for the actual forward/backward pass.

    For batchwise losses such as LogSmoothAP and soft-rank Spearman, ordinary
    gradient accumulation over independent microbatch losses is not equivalent
    to a full logical-batch loss. Therefore, this class uses cached-logit
    gradient replay by default:

        1. Run the logical batch in microbatches under no_grad and cache logits.
        2. Compute the true batchwise loss over the full logical-batch logits.
        3. Differentiate that loss with respect to the logits.
        4. Replay the model in microbatches and inject the cached logit-gradient.

    This preserves the 3072-sample LogSmoothAP / Spearman comparison set while
    keeping activation memory near the microbatch size.

    For independent losses such as BCEWithLogits and MSE, this class uses direct
    microbatch backpropagation with exact full-batch mean scaling.

    Notes
    -----
    - The DataLoader batch_size remains the logical batch size. Batches are kept
      on CPU and only microbatches are moved to the GPU.
    - Sampling calibration is intentionally unchanged from PepCL; the logical
      batch size is still what determines per-allele quotas.
    - The model is called without a `label` key, matching the original PepCL code
      after `lbl = batch.pop('label')`.
    """

    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,

        steps: int = 200,
        output_dir: str = None,
        save_every_n_steps: int = 100,
        seed: int = 42,
        num_workers: int = 8,
        collate_fn=None,

        # Logical stream batch sizes. These should usually match the full PepCL
        # batch sizes you are trying to mimic, e.g. 3072.
        num_ms_pos: int = 50,
        ms_neg_ratio: float = 1.0,
        num_nonms_pos: int = 50,
        nonms_neg_ratio: float = 1.0,
        ms_batch_size: int = 3072,
        nonms_batch_size: int = 3072,

        # Physical autograd microbatch sizes. These are the memory-control knobs.
        ms_microbatch_size: int = 512,
        nonms_microbatch_size: int = 512,

        auto_calibrate: bool = True,
        calibration_steps: int = 1,  # kept for backward compatibility

        continuous_target_col: str = "ba_value",
        continuous_ft_sampling_mode: str = "allele_aware",  # "allele_aware" or "pooled"
        pooled_ft_num_pos: int = None,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        update_domain_embeddings: bool = False,

        use_replay: bool = True,
        primary_task_id: int = 1,
        sequential_backprop: bool = True,

        ms_loss_type: str = "logsmoothap",
        ms_loss_alpha: float = 10.0,
        ms_lambda: float = 1.0,
        nonms_loss_type: str = "logsmoothap",
        nonms_loss_alpha: float = 10.0,
        ft_lambda: float = 1.0,
        dr_lambda: float = 0.0,
        dr_loss_type: str = "mse",

        # Micro/caching behavior.
        # "auto": cached replay for batchwise losses, direct microbatch for independent losses.
        # "cache": force cached replay for all supported losses.
        # "direct": direct microbatch losses; this is NOT faithful for batchwise losses.
        micro_strategy: str = "auto",
        batchwise_loss_types=None,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        preserve_cache_rng_state: bool = True,
        cache_forward_train_mode: bool = True,
        empty_cache_between_streams: bool = False,
        fail_on_nonfinite_loss: bool = True,
    ):
        self.model = model.to(device)
        self.df = train_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.device_type = self.device.type
        self.steps = steps
        self.output_dir = output_dir
        self.save_every_n_steps = save_every_n_steps
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        self.ms_loss_type = str(ms_loss_type).lower()
        self.nonms_loss_type = str(nonms_loss_type).lower()

        self.continuous_ft_loss_types = {"spearman", "mse"}
        self.ft_is_continuous = self.nonms_loss_type in self.continuous_ft_loss_types

        self.continuous_target_col = continuous_target_col
        self.continuous_ft_sampling_mode = continuous_ft_sampling_mode
        self.pooled_ft_num_pos = pooled_ft_num_pos

        if self.continuous_ft_sampling_mode not in {"allele_aware", "pooled"}:
            raise ValueError("continuous_ft_sampling_mode must be one of: {'allele_aware', 'pooled'}")

        if self.continuous_ft_sampling_mode == "pooled" and not self.ft_is_continuous:
            raise ValueError(
                "continuous_ft_sampling_mode='pooled' is only supported when "
                "nonms_loss_type is continuous, i.e. 'spearman' or 'mse'."
            )

        if self.ft_is_continuous and self.continuous_target_col not in self.df.columns:
            raise ValueError(
                f"Continuous FT requested because nonms_loss_type='{self.nonms_loss_type}', "
                f"but continuous_target_col='{self.continuous_target_col}' is not in train_df."
            )

        self.ms_bs, self.ft_bs = int(ms_batch_size), int(nonms_batch_size)
        self.ms_micro_bs = int(ms_microbatch_size or ms_batch_size)
        self.ft_micro_bs = int(nonms_microbatch_size or nonms_batch_size)

        if self.ms_micro_bs <= 0 or self.ft_micro_bs <= 0:
            raise ValueError("Microbatch sizes must be positive integers.")
        if self.ms_micro_bs > self.ms_bs:
            warnings.warn("ms_microbatch_size is larger than ms_batch_size; clipping to ms_batch_size.")
            self.ms_micro_bs = self.ms_bs
        if self.ft_micro_bs > self.ft_bs:
            warnings.warn("nonms_microbatch_size is larger than nonms_batch_size; clipping to nonms_batch_size.")
            self.ft_micro_bs = self.ft_bs

        self.auto_calibrate = auto_calibrate
        self.calibration_steps = calibration_steps

        self.ms_cfg = {
            "n_pos": num_ms_pos,
            "ratio": ms_neg_ratio,
            "bs": self.ms_bs,
            "micro_bs": self.ms_micro_bs,
            "q_pos_by_allele": None,
            "sampling_summary": None,
            "sampling_mode": "allele_aware",
        }
        self.ft_cfg = {
            "n_pos": num_nonms_pos,
            "ratio": nonms_neg_ratio,
            "bs": self.ft_bs,
            "micro_bs": self.ft_micro_bs,
            "q_pos_by_allele": None,
            "sampling_summary": None,
            "sampling_mode": self.continuous_ft_sampling_mode if self.ft_is_continuous else "allele_aware",
            "pooled_ft_num_pos": self.pooled_ft_num_pos,
        }

        if self.ft_is_continuous:
            self.base_dataset = PeptideMHCDataset(
                self.df,
                self.tokenizer,
                include_domain_id=True,
                feature_names=[self.continuous_target_col],
            )
        else:
            self.base_dataset = PeptideMHCDataset(
                self.df,
                self.tokenizer,
                include_domain_id=True,
            )

        self.use_replay = use_replay
        self.primary_task_id = primary_task_id
        self.seq_backprop = sequential_backprop
        if not self.seq_backprop:
            warnings.warn(
                "PepCL_Micro was designed to backpropagate streams sequentially and step once. "
                "sequential_backprop=False was requested, but additive stream gradients are still "
                "accumulated stream-by-stream to avoid storing both stream graphs."
            )
        self.update_domain_embeddings = update_domain_embeddings
        self.dr_lambda = dr_lambda
        self.dr_loss_type = dr_loss_type
        self.ms_lambda, self.ft_lambda = ms_lambda, ft_lambda

        self.encoder_lr, self.decoder_lr = encoder_lr, decoder_lr
        self.weight_decay, self.warmup_ratio = weight_decay, warmup_ratio

        self.criterion_ms = self._get_criterion(self.ms_loss_type, ms_loss_alpha)
        self.criterion_ft = self._get_criterion(self.nonms_loss_type, nonms_loss_alpha)

        self.teacher = None
        if self.use_replay and self.dr_lambda > 0:
            self.teacher = copy.deepcopy(model).to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.micro_strategy = str(micro_strategy).lower()
        if self.micro_strategy not in {"auto", "cache", "direct"}:
            raise ValueError("micro_strategy must be one of: {'auto', 'cache', 'direct'}")

        if batchwise_loss_types is None:
            batchwise_loss_types = {"logsmoothap", "spearman"}
        self.batchwise_loss_types = {str(x).lower() for x in batchwise_loss_types}

        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype
        self.preserve_cache_rng_state = bool(preserve_cache_rng_state)
        self.cache_forward_train_mode = bool(cache_forward_train_mode)
        self.empty_cache_between_streams = bool(empty_cache_between_streams)
        self.fail_on_nonfinite_loss = bool(fail_on_nonfinite_loss)

        torch.manual_seed(seed)
        np.random.seed(seed)

    # ---------------------------------------------------------------------
    # Losses
    # ---------------------------------------------------------------------
    class _LogSmoothAPScore(nn.Module):
        def __init__(self, alpha=10.0, epsilon=1e-8):
            super().__init__()
            self.alpha, self.epsilon = alpha, epsilon

        def forward(self, scores, labels):
            y = labels.float()
            P = y.sum().clamp(min=1.0)
            diff = scores.unsqueeze(0) - scores.unsqueeze(1)
            S = torch.sigmoid(self.alpha * diff)
            num = S.matmul(y)
            den = S.sum(dim=1).clamp(min=self.epsilon)
            soft_prec = num / den
            softap = (soft_prec * y).sum() / P
            return torch.log(1.0 / (softap + self.epsilon))

    @staticmethod
    def _soft_rank(x: torch.Tensor, tau: float = 1.0):
        """
        Differentiable rank based on pairwise sigmoid.
        r_i = 1 + Σ_j σ((x_j - x_i) / τ)
        τ -> 0 gives hard ranks.
        x : (..., L)
        returns (..., L) in [1, L]
        """
        x_col = x.unsqueeze(-1)
        x_row = x.unsqueeze(-2)
        P = torch.sigmoid((x_row - x_col) / tau)
        return 1 + P.sum(-1)

    class _SoftSpearmanLoss(nn.Module):
        def __init__(self, tau: float = 0.1):
            super().__init__()
            self.tau = tau

        def forward(self, pred: torch.Tensor, target: torch.Tensor):
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)

            r_pred = PepCL_Micro._soft_rank(pred, tau=self.tau)
            r_target = PepCL_Micro._soft_rank(target, tau=1e-4)

            r_pred = r_pred - r_pred.mean(-1, keepdim=True)
            r_target = r_target - r_target.mean(-1, keepdim=True)

            num = (r_pred * r_target).sum(-1)
            den = torch.norm(r_pred, dim=-1) * torch.norm(r_target, dim=-1) + 1e-8
            rho = num / den
            return (1 - rho).mean()

    class _TargetOverrideDataset(Dataset):
        """
        Wraps the parent PeptideMHCDataset so a stream can override:
            - domain_id
            - label target

        For classification streams:
            label remains the binary label from the parent dataset / dataframe.

        For continuous FT streams:
            label becomes peptide_{continuous_target_col}, e.g. peptide_ba_value,
            or falls back to df[continuous_target_col].
        """
        def __init__(
            self,
            parent_ds,
            df,
            domain_id,
            continuous_target: bool = False,
            continuous_target_col: str = "ba_value",
        ):
            self.parent_ds = parent_ds
            self.df = df
            self.domain_id = domain_id
            self.continuous_target = continuous_target
            self.continuous_target_col = continuous_target_col
            self.item_continuous_key = f"peptide_{continuous_target_col}"

        def __len__(self):
            return len(self.parent_ds)

        def __getitem__(self, idx):
            item = self.parent_ds[idx]
            item = dict(item)
            item["domain_id"] = torch.tensor(self.domain_id, dtype=torch.long)

            if self.continuous_target:
                if self.item_continuous_key in item:
                    target = item[self.item_continuous_key]
                    if isinstance(target, torch.Tensor):
                        item["label"] = target.float()
                    else:
                        item["label"] = torch.tensor(target, dtype=torch.float)
                else:
                    target = self.df.loc[idx, self.continuous_target_col]
                    item["label"] = torch.tensor(target, dtype=torch.float)

            return item

    class _InfiniteAlleleSampler(Sampler):
        """
        Infinite allele-aware sampler.

        Manual mode:
            Uses scalar n_pos per allele per sampler cycle.

        Auto mode:
            Uses q_pos_by_allele, a dict mapping allele -> positive quota per sampler cycle.

        Negative behavior:
            neg_ratio = 1.0 means sample equal number of negatives per selected positives.
            neg_ratio = 0.0 means sample no negatives.
            If an allele has no negatives, this is allowed and no negatives are sampled for that allele.
        """
        def __init__(self, df, domain_id, n_pos, neg_ratio, seed, q_pos_by_allele=None):
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.n_pos = n_pos
            self.neg_ratio = float(neg_ratio)
            self.q_pos_by_allele = q_pos_by_allele

            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

            self.pools = {0: {}, 1: {}}
            subset = df[df["domain_id"] == domain_id]
            if subset.empty:
                self.alleles = []
                return

            for (lbl, allele), group in subset.groupby(["label", "allele"]):
                lbl_int = int(lbl)
                if lbl_int not in self.pools:
                    continue
                idx = group.index.to_numpy()
                self.rng.shuffle(idx)
                self.pools[lbl_int][allele] = idx

            positive_alleles = sorted(list(self.pools[1].keys()))

            if self.q_pos_by_allele is not None:
                self.q_pos_by_allele = {
                    al: int(q) for al, q in self.q_pos_by_allele.items() if int(q) > 0
                }
                self.alleles = [
                    al for al in positive_alleles if self.q_pos_by_allele.get(al, 0) > 0
                ]
            else:
                self.alleles = positive_alleles

        def __iter__(self):
            if not self.alleles:
                return

            while True:
                indices = []

                for al in self.alleles:
                    pos_data = self.pools[1].get(al, [])
                    if len(pos_data) == 0:
                        continue

                    if self.q_pos_by_allele is not None:
                        requested_pos = int(self.q_pos_by_allele.get(al, 0))
                    else:
                        requested_pos = int(self.n_pos)

                    if requested_pos <= 0:
                        continue

                    p_take = min(len(pos_data), requested_pos)
                    if p_take <= 0:
                        continue

                    sel_pos = self.rng.choice(pos_data, size=p_take, replace=False)
                    indices.extend(sel_pos)

                    neg_data = self.pools[0].get(al, [])
                    n_req = int(len(sel_pos) * self.neg_ratio)

                    if n_req > 0 and len(neg_data) > 0:
                        n_take = min(len(neg_data), n_req)
                        if n_take > 0:
                            sel_neg = self.rng.choice(neg_data, size=n_take, replace=False)
                            indices.extend(sel_neg)

                if len(indices) == 0:
                    return

                self.rng.shuffle(indices)
                for i in indices:
                    yield i

    class _InfinitePooledFTSampler(Sampler):
        """
        Infinite global pooled sampler for continuous FT/BA mode.
        """
        def __init__(self, df, domain_id, pooled_num_pos, neg_ratio, batch_size, seed):
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.neg_ratio = float(neg_ratio)
            self.batch_size = int(batch_size)

            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

            subset = df[df["domain_id"] == domain_id].copy()
            if subset.empty:
                self.pos_pool = np.array([], dtype=int)
                self.neg_pool = np.array([], dtype=int)
                self.pooled_num_pos = 0
                return

            label_int = subset["label"].astype(int)
            self.pos_pool = subset.index[label_int == 1].to_numpy()
            self.neg_pool = subset.index[label_int == 0].to_numpy()

            if pooled_num_pos is None:
                pooled_num_pos = int(self.batch_size / (1.0 + self.neg_ratio))
                pooled_num_pos = max(1, pooled_num_pos)

            self.pooled_num_pos = int(pooled_num_pos)
            self.rng.shuffle(self.pos_pool)
            self.rng.shuffle(self.neg_pool)

        def __iter__(self):
            if len(self.pos_pool) == 0 or self.pooled_num_pos <= 0:
                return

            while True:
                indices = []

                p_take = min(self.pooled_num_pos, len(self.pos_pool))
                if p_take > 0:
                    sel_pos = self.rng.choice(self.pos_pool, size=p_take, replace=False)
                    indices.extend(sel_pos)

                    n_req = int(len(sel_pos) * self.neg_ratio)
                    if n_req > 0 and len(self.neg_pool) > 0:
                        replace = len(self.neg_pool) < n_req
                        sel_neg = self.rng.choice(self.neg_pool, size=n_req, replace=replace)
                        indices.extend(sel_neg)

                if len(indices) == 0:
                    return

                self.rng.shuffle(indices)
                for i in indices:
                    yield i

    # ---------------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------------
    def _get_criterion(self, t, alpha):
        t = str(t).lower()

        if t == "logsmoothap":
            return self._LogSmoothAPScore(alpha=alpha)
        if t == "spearman":
            return self._SoftSpearmanLoss(tau=alpha)
        if t == "mse":
            return nn.MSELoss()
        if t in {"bce", "bcewithlogits", "bcewithlogitsloss"}:
            return nn.BCEWithLogitsLoss()
        return nn.BCEWithLogitsLoss()

    def _get_dr_loss(self, student_logits, teacher_logits):
        if self.dr_loss_type == "mse":
            return F.mse_loss(student_logits, teacher_logits)
        return F.mse_loss(student_logits, teacher_logits)

    def _get_optimizer(self):
        e, d = [], []
        for n, p in self.model.named_parameters():
            if "domain_embedder" in n and not self.update_domain_embeddings:
                p.requires_grad = False
                continue
            if not p.requires_grad:
                continue
            if any(x in n for x in ["embedder", "conformer", "transformer"]):
                e.append(p)
            else:
                d.append(p)

        return torch.optim.AdamW(
            [
                {"params": e, "lr": self.encoder_lr},
                {"params": d, "lr": self.decoder_lr},
            ],
            weight_decay=self.weight_decay,
        )

    def _autocast_context(self):
        enabled = bool(self.use_amp and self.device_type == "cuda")
        if enabled:
            return autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=True)
        return nullcontext()

    def _make_scaler(self):
        enabled = bool(self.use_amp and self.device_type == "cuda")
        try:
            return GradScaler(device=self.device_type, enabled=enabled)
        except TypeError:
            try:
                return GradScaler(self.device_type, enabled=enabled)
            except TypeError:
                return GradScaler(enabled=enabled)

    def _capture_rng_state(self):
        if not self.preserve_cache_rng_state:
            return None
        state = {"cpu": torch.get_rng_state()}
        if self.device_type == "cuda" and torch.cuda.is_available():
            state["cuda_all"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        if state is None:
            return
        torch.set_rng_state(state["cpu"])
        if "cuda_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_all"])

    @staticmethod
    def _infer_batch_size(obj):
        if isinstance(obj, torch.Tensor) and obj.dim() > 0:
            return int(obj.shape[0])
        if isinstance(obj, dict):
            preferred = ["peptide", "label", "mhc_list", "peptide_mask", "mhc_mask_list"]
            for k in preferred:
                if k in obj and isinstance(obj[k], torch.Tensor) and obj[k].dim() > 0:
                    return int(obj[k].shape[0])
            for v in obj.values():
                n = PepCL_Micro._infer_batch_size(v)
                if n is not None:
                    return n
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            return len(obj)
        return None

    @staticmethod
    def _slice_obj(obj, start, end, batch_size):
        if isinstance(obj, torch.Tensor):
            if obj.dim() > 0 and int(obj.shape[0]) == int(batch_size):
                return obj[start:end]
            return obj
        if isinstance(obj, dict):
            return {k: PepCL_Micro._slice_obj(v, start, end, batch_size) for k, v in obj.items()}
        if isinstance(obj, list):
            if len(obj) == int(batch_size):
                return obj[start:end]
            return obj
        if isinstance(obj, tuple):
            if len(obj) == int(batch_size):
                return obj[start:end]
            return obj
        return obj

    def _move_obj_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_obj_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_obj_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_obj_to_device(v) for v in obj)
        return obj

    def _iter_microbatches(self, batch_cpu, microbatch_size):
        n = self._infer_batch_size(batch_cpu)
        if n is None:
            raise ValueError("Could not infer batch size from batch dictionary.")
        for start in range(0, n, microbatch_size):
            end = min(start + microbatch_size, n)
            mb_cpu = self._slice_obj(batch_cpu, start, end, n)
            mb_gpu = self._move_obj_to_device(mb_cpu)
            yield mb_gpu, start, end

    @staticmethod
    def _is_bce_loss_type(loss_type: str):
        return str(loss_type).lower() in {"bce", "bcewithlogits", "bcewithlogitsloss"}

    @staticmethod
    def _is_mse_loss_type(loss_type: str):
        return str(loss_type).lower() == "mse"

    def _use_cached_replay_for_loss(self, loss_type: str):
        loss_type = str(loss_type).lower()
        if self.micro_strategy == "cache":
            return True
        if self.micro_strategy == "direct":
            return False
        return loss_type in self.batchwise_loss_types

    def _check_finite(self, loss_tensor, name):
        if self.fail_on_nonfinite_loss and not torch.isfinite(loss_tensor).all().item():
            raise FloatingPointError(f"Non-finite {name}: {loss_tensor.detach().cpu().item()}")

    # ---------------------------------------------------------------------
    # Sampling calibration: intentionally unchanged from PepCL
    # ---------------------------------------------------------------------
    @staticmethod
    def _integer_quota_allocation(weights, caps, total, ensure_all_if_possible=True):
        """
        Allocate an integer total across alleles according to weights, respecting caps.
        """
        weights = np.asarray(weights, dtype=float)
        caps = np.asarray(caps, dtype=int)
        total = int(total)

        q = np.zeros_like(caps, dtype=int)
        valid = caps > 0
        if total <= 0 or not np.any(valid):
            return q

        max_total = int(min(total, caps.sum()))
        if max_total <= 0:
            return q

        w = np.zeros_like(weights, dtype=float)
        if weights[valid].sum() > 0:
            w[valid] = weights[valid] / weights[valid].sum()
        else:
            w[valid] = 1.0 / valid.sum()

        remaining_total = max_total

        if ensure_all_if_possible and remaining_total >= valid.sum():
            q[valid] = 1
            remaining_total -= int(valid.sum())

        if remaining_total <= 0:
            return q

        while remaining_total > 0:
            capacity = caps - q
            can_receive = capacity > 0
            if not np.any(can_receive):
                break

            w_active = np.zeros_like(w, dtype=float)
            if w[can_receive].sum() > 0:
                w_active[can_receive] = w[can_receive] / w[can_receive].sum()
            else:
                w_active[can_receive] = 1.0 / can_receive.sum()

            raw = remaining_total * w_active
            add = np.floor(raw).astype(int)
            add = np.minimum(add, capacity)

            if add.sum() > 0:
                q += add
                remaining_total -= int(add.sum())
                if remaining_total <= 0:
                    break

            capacity = caps - q
            can_receive = capacity > 0
            if not np.any(can_receive):
                break

            frac = raw - np.floor(raw)
            frac[~can_receive] = -1.0
            order = np.argsort(-frac)
            added_any = False

            for idx in order:
                if remaining_total <= 0:
                    break
                if capacity[idx] <= 0:
                    continue
                q[idx] += 1
                remaining_total -= 1
                added_any = True

            if not added_any:
                break

        return q

    def _compute_auto_domain_sampling(self, domain_id, batch_size, neg_ratio):
        """
        Data-adaptive allele-aware sampling calibration.
        Uses the logical batch size, not the physical microbatch size.
        """
        if neg_ratio < 0:
            raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

        sub = self.df[self.df["domain_id"] == domain_id].copy()
        if sub.empty:
            return {}, pd.DataFrame()

        sub["_label_int_for_sampling"] = sub["label"].astype(int)

        counts = (
            sub.groupby(["allele", "_label_int_for_sampling"])
            .size()
            .unstack(fill_value=0)
        )

        n_pos = counts[1] if 1 in counts.columns else pd.Series(0, index=counts.index)
        n_neg = counts[0] if 0 in counts.columns else pd.Series(0, index=counts.index)

        summary = pd.DataFrame({
            "allele": counts.index,
            "n_pos": n_pos.astype(int).to_numpy(),
            "n_neg": n_neg.astype(int).to_numpy(),
        })
        summary = summary[summary["n_pos"] > 0].reset_index(drop=True)

        if summary.empty:
            return {}, summary

        A = len(summary)
        P = summary["n_pos"].astype(float).to_numpy()
        caps = summary["n_pos"].astype(int).to_numpy()

        target_pos = int(batch_size / (1.0 + float(neg_ratio)))
        target_pos = max(1, target_pos)

        uniform = np.ones(A, dtype=float) / A
        prevalence = P / P.sum()

        denom = np.square(P).sum()
        if denom <= 0:
            A_eff = float(A)
        else:
            A_eff = float((P.sum() ** 2) / denom)

        balance_score = float(A_eff / A)
        balance_score = max(0.0, min(1.0, balance_score))

        weights = balance_score * uniform + (1.0 - balance_score) * prevalence
        weights = weights / weights.sum()

        q_pos = self._integer_quota_allocation(
            weights=weights,
            caps=caps,
            total=target_pos,
            ensure_all_if_possible=False,
        )

        q_pos_by_allele = {
            al: int(q)
            for al, q in zip(summary["allele"].tolist(), q_pos)
            if int(q) > 0
        }

        summary["allele_weight"] = weights
        summary["q_pos"] = q_pos.astype(int)
        summary["expected_q_neg_if_available"] = np.floor(
            summary["q_pos"].to_numpy() * float(neg_ratio)
        ).astype(int)
        summary["reuse_pressure_pos"] = summary["q_pos"] / summary["n_pos"].clip(lower=1)
        summary["has_negatives"] = summary["n_neg"] > 0

        summary.attrs["target_pos"] = target_pos
        summary.attrs["n_alleles"] = A
        summary.attrs["effective_n_alleles"] = A_eff
        summary.attrs["balance_score"] = balance_score

        return q_pos_by_allele, summary

    def _compute_pooled_ft_summary(self):
        sub = self.df[self.df["domain_id"] == 1].copy()
        if sub.empty:
            return pd.DataFrame()

        sub["_label_int_for_sampling"] = sub["label"].astype(int)

        n_pos_total = int((sub["_label_int_for_sampling"] == 1).sum())
        n_neg_total = int((sub["_label_int_for_sampling"] == 0).sum())
        n_alleles_pos = int(sub.loc[sub["_label_int_for_sampling"] == 1, "allele"].nunique())
        n_alleles_all = int(sub["allele"].nunique())

        if self.pooled_ft_num_pos is None:
            pooled_pos = int(self.ft_cfg["bs"] / (1.0 + float(self.ft_cfg["ratio"])))
            pooled_pos = max(1, pooled_pos)
        else:
            pooled_pos = int(self.pooled_ft_num_pos)

        pooled_neg = int(pooled_pos * float(self.ft_cfg["ratio"]))

        summary = pd.DataFrame({
            "mode": ["pooled"],
            "n_pos_total": [n_pos_total],
            "n_neg_total": [n_neg_total],
            "n_alleles_pos": [n_alleles_pos],
            "n_alleles_all": [n_alleles_all],
            "pooled_ft_num_pos": [pooled_pos],
            "expected_pooled_num_neg": [pooled_neg],
            "pos_reuse_pressure": [pooled_pos / max(n_pos_total, 1)],
        })

        summary.attrs["pooled_ft_num_pos"] = pooled_pos
        summary.attrs["expected_pooled_num_neg"] = pooled_neg
        return summary

    def _auto_calibrate(self):
        ms_q, ms_summary = self._compute_auto_domain_sampling(
            domain_id=0,
            batch_size=self.ms_cfg["bs"],
            neg_ratio=self.ms_cfg["ratio"],
        )
        self.ms_cfg["q_pos_by_allele"] = ms_q
        self.ms_cfg["sampling_summary"] = ms_summary

        if self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
            ft_summary = self._compute_pooled_ft_summary()
            self.ft_cfg["q_pos_by_allele"] = None
            self.ft_cfg["sampling_summary"] = ft_summary
        else:
            ft_q, ft_summary = self._compute_auto_domain_sampling(
                domain_id=1,
                batch_size=self.ft_cfg["bs"],
                neg_ratio=self.ft_cfg["ratio"],
            )
            self.ft_cfg["q_pos_by_allele"] = ft_q
            self.ft_cfg["sampling_summary"] = ft_summary

        def _summary_line(name, summary, cfg):
            if summary is None or summary.empty:
                return f"  {name}: no eligible positive alleles"

            if "mode" in summary.columns and summary["mode"].iloc[0] == "pooled":
                row = summary.iloc[0]
                return (
                    f"  {name}: pooled continuous FT, "
                    f"alleles_pos={int(row['n_alleles_pos'])}, alleles_all={int(row['n_alleles_all'])}, "
                    f"pos_pool={int(row['n_pos_total'])}, neg_pool={int(row['n_neg_total'])}, "
                    f"pooled_pos/cycle={int(row['pooled_ft_num_pos'])}, "
                    f"expected_neg/cycle={int(row['expected_pooled_num_neg'])}, "
                    f"pos_reuse={float(row['pos_reuse_pressure']):.4f}, "
                    f"neg_ratio={cfg['ratio']}, logical_bs={cfg['bs']}, micro_bs={cfg['micro_bs']}"
                )

            target_pos = summary.attrs.get("target_pos", None)
            A = summary.attrs.get("n_alleles", len(summary))
            A_eff = summary.attrs.get("effective_n_alleles", np.nan)
            balance_score = summary.attrs.get("balance_score", np.nan)

            q_nonzero = summary.loc[summary["q_pos"] > 0, "q_pos"]
            reuse = summary.loc[summary["q_pos"] > 0, "reuse_pressure_pos"]

            q_med = float(q_nonzero.median()) if len(q_nonzero) else 0.0
            q_min = int(q_nonzero.min()) if len(q_nonzero) else 0
            q_max = int(q_nonzero.max()) if len(q_nonzero) else 0
            reuse_med = float(reuse.median()) if len(reuse) else 0.0
            reuse_max = float(reuse.max()) if len(reuse) else 0.0

            return (
                f"  {name}: alleles={A}, A_eff={A_eff:.2f}, balance={balance_score:.3f}, "
                f"target_pos/cycle={target_pos}, q_pos median={q_med:.1f} "
                f"[{q_min}, {q_max}], reuse median={reuse_med:.4f}, reuse max={reuse_max:.4f}, "
                f"neg_ratio={cfg['ratio']}, logical_bs={cfg['bs']}, micro_bs={cfg['micro_bs']}"
            )

        print("[PepCL_Micro] Auto-calibrated sampling:")
        print(_summary_line("MS", self.ms_cfg["sampling_summary"], self.ms_cfg))
        print(_summary_line("FT", self.ft_cfg["sampling_summary"], self.ft_cfg))
        print("")

    def _build_loader(self, domain_id, cfg, seed):
        continuous_target = domain_id == 1 and self.ft_is_continuous

        ds = self._TargetOverrideDataset(
            parent_ds=self.base_dataset,
            df=self.df,
            domain_id=domain_id,
            continuous_target=continuous_target,
            continuous_target_col=self.continuous_target_col,
        )

        if (
            domain_id == 1
            and self.ft_is_continuous
            and cfg.get("sampling_mode", "allele_aware") == "pooled"
        ):
            sampler = self._InfinitePooledFTSampler(
                self.df,
                domain_id=domain_id,
                pooled_num_pos=cfg.get("pooled_ft_num_pos", None),
                neg_ratio=cfg["ratio"],
                batch_size=cfg["bs"],
                seed=seed,
            )
        else:
            sampler = self._InfiniteAlleleSampler(
                self.df,
                domain_id,
                cfg["n_pos"],
                cfg["ratio"],
                seed,
                q_pos_by_allele=cfg.get("q_pos_by_allele", None),
            )

        return DataLoader(
            ds,
            batch_size=cfg["bs"],  # logical batch size, not microbatch size
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=(self.device_type == "cuda"),
            persistent_workers=(self.num_workers > 0),
        )

    # ---------------------------------------------------------------------
    # Microbatch backward implementations
    # ---------------------------------------------------------------------
    def _cached_batchwise_backward(
        self,
        batch_cpu,
        labels_cpu,
        criterion,
        loss_type,
        loss_weight,
        microbatch_size,
        scaler,
        teacher=None,
        dr_lambda=0.0,
        stream_name="stream",
    ):
        """
        Full-logical-batch loss with microbatch activation replay.

        This is the faithful path for LogSmoothAP and soft-rank Spearman.
        It also works for BCE/MSE if forced, but direct microbatching is more
        natural for independent losses.
        """
        labels = labels_cpu.to(self.device, non_blocking=True).float().view(-1)
        n = int(labels.numel())

        was_training = self.model.training
        if self.cache_forward_train_mode:
            self.model.train()
        else:
            self.model.eval()

        logits_parts = []
        teacher_parts = []
        rng_states = []

        with torch.no_grad():
            for mb, _, _ in self._iter_microbatches(batch_cpu, microbatch_size):
                rng_states.append(self._capture_rng_state())
                with self._autocast_context():
                    z = self.model(mb).view(-1)
                logits_parts.append(z.detach())

                if teacher is not None and dr_lambda > 0:
                    teacher.eval()
                    with self._autocast_context():
                        t = teacher(mb).view(-1)
                    teacher_parts.append(t.detach())

        if was_training:
            self.model.train()
        else:
            self.model.eval()

        logits_full = torch.cat(logits_parts, dim=0).float()
        if logits_full.numel() != n:
            raise RuntimeError(
                f"{stream_name}: cached logits count {logits_full.numel()} != labels count {n}."
            )

        logits_leaf = logits_full.detach().requires_grad_(True)
        loss_main = criterion(logits_leaf, labels) * float(loss_weight)
        loss_total = loss_main
        loss_dr = None

        if teacher is not None and dr_lambda > 0:
            teacher_full = torch.cat(teacher_parts, dim=0).float()
            if teacher_full.numel() != n:
                raise RuntimeError(
                    f"{stream_name}: teacher logits count {teacher_full.numel()} != labels count {n}."
                )
            loss_dr = self._get_dr_loss(logits_leaf, teacher_full) * float(dr_lambda)
            loss_total = loss_total + loss_dr

        self._check_finite(loss_total, f"{stream_name} cached loss")
        grad_logits = torch.autograd.grad(loss_total, logits_leaf, retain_graph=False)[0].detach()

        # Replay the model with graph in small chunks and inject the full-batch logit gradient.
        self.model.train() if was_training else self.model.eval()
        chunk_idx = 0
        for mb, start, end in self._iter_microbatches(batch_cpu, microbatch_size):
            if self.preserve_cache_rng_state:
                self._restore_rng_state(rng_states[chunk_idx])
            grad_chunk = grad_logits[start:end]
            chunk_idx += 1

            with self._autocast_context():
                logits_chunk = self.model(mb).view(-1)
                # This proxy scalar has the desired vector-Jacobian product:
                # d/dtheta sum_i grad_i * logit_i(theta)
                proxy = (logits_chunk.float() * grad_chunk).sum()

            scaler.scale(proxy).backward()

        return {
            "loss": float(loss_total.detach().cpu().item()),
            "main_loss": float(loss_main.detach().cpu().item()),
            "dr_loss": 0.0 if loss_dr is None else float(loss_dr.detach().cpu().item()),
            "mode": "cached",
        }

    def _direct_microbatch_backward(
        self,
        batch_cpu,
        labels_cpu,
        criterion,
        loss_type,
        loss_weight,
        microbatch_size,
        scaler,
        teacher=None,
        dr_lambda=0.0,
        stream_name="stream",
    ):
        """
        Direct microbatch backprop for independent losses.

        Exact full-logical-batch mean scaling is used for BCE/MSE. This should
        not be used for LogSmoothAP/Spearman if your goal is to reproduce the
        logical-batch 3072 ranking objective.
        """
        loss_type = str(loss_type).lower()
        labels_full = labels_cpu.to(self.device, non_blocking=True).float().view(-1)
        n_total = int(labels_full.numel())
        total_loss_value = 0.0
        total_main_value = 0.0
        total_dr_value = 0.0

        for mb, start, end in self._iter_microbatches(batch_cpu, microbatch_size):
            labels = labels_full[start:end]
            n_chunk = int(labels.numel())
            weight = float(n_chunk) / float(max(n_total, 1))

            with self._autocast_context():
                logits = self.model(mb).view(-1)
                loss_main_unweighted = criterion(logits, labels)
                loss_main = loss_main_unweighted * float(loss_weight)
                loss_total = loss_main
                loss_dr = None

                if teacher is not None and dr_lambda > 0:
                    teacher.eval()
                    with torch.no_grad():
                        t_logits = teacher(mb).view(-1)
                    loss_dr = self._get_dr_loss(logits, t_logits) * float(dr_lambda)
                    loss_total = loss_total + loss_dr

                # criterion is mean-reduced, so scale each chunk by n_chunk / n_total
                scaled_loss = loss_total * weight

            self._check_finite(scaled_loss, f"{stream_name} direct scaled loss")
            scaler.scale(scaled_loss).backward()

            total_loss_value += float(loss_total.detach().cpu().item()) * weight
            total_main_value += float(loss_main.detach().cpu().item()) * weight
            if loss_dr is not None:
                total_dr_value += float(loss_dr.detach().cpu().item()) * weight

        return {
            "loss": total_loss_value,
            "main_loss": total_main_value,
            "dr_loss": total_dr_value,
            "mode": "direct",
        }

    def _stream_backward(
        self,
        batch_cpu,
        labels_cpu,
        criterion,
        loss_type,
        loss_weight,
        microbatch_size,
        scaler,
        teacher=None,
        dr_lambda=0.0,
        stream_name="stream",
    ):
        if self._use_cached_replay_for_loss(loss_type):
            return self._cached_batchwise_backward(
                batch_cpu=batch_cpu,
                labels_cpu=labels_cpu,
                criterion=criterion,
                loss_type=loss_type,
                loss_weight=loss_weight,
                microbatch_size=microbatch_size,
                scaler=scaler,
                teacher=teacher,
                dr_lambda=dr_lambda,
                stream_name=stream_name,
            )

        if str(loss_type).lower() in self.batchwise_loss_types:
            warnings.warn(
                f"{stream_name}: using direct microbatching for batchwise loss '{loss_type}'. "
                "This is intentionally allowed but is not faithful to the full logical-batch objective."
            )

        return self._direct_microbatch_backward(
            batch_cpu=batch_cpu,
            labels_cpu=labels_cpu,
            criterion=criterion,
            loss_type=loss_type,
            loss_weight=loss_weight,
            microbatch_size=microbatch_size,
            scaler=scaler,
            teacher=teacher,
            dr_lambda=dr_lambda,
            stream_name=stream_name,
        )

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    def train(self):
        if self.auto_calibrate:
            self._auto_calibrate()
        else:
            if self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
                self.ft_cfg["sampling_summary"] = self._compute_pooled_ft_summary()

            print(
                f"[PepCL_Micro] Manual mode: "
                f"MS n_pos={self.ms_cfg['n_pos']}, MS neg_ratio={self.ms_cfg['ratio']}, "
                f"MS logical_bs={self.ms_cfg['bs']}, MS micro_bs={self.ms_cfg['micro_bs']} | "
                f"FT n_pos={self.ft_cfg['n_pos']}, FT neg_ratio={self.ft_cfg['ratio']}, "
                f"FT logical_bs={self.ft_cfg['bs']}, FT micro_bs={self.ft_cfg['micro_bs']} | "
                f"FT continuous={self.ft_is_continuous}, "
                f"FT sampling_mode={self.ft_cfg['sampling_mode']}\n"
            )

        ms_iter = iter(self._build_loader(0, self.ms_cfg, self.seed))
        ft_iter = iter(self._build_loader(1, self.ft_cfg, self.seed + 1))

        optimizer = self._get_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(self.steps * self.warmup_ratio),
            self.steps,
        )
        scaler = self._make_scaler()

        metrics = collections.defaultdict(list)
        pbar = tqdm(range(self.steps), desc="PepCL_Micro")

        for step in pbar:
            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            l_ms_v, l_ft_v, l_dr_v = 0.0, 0.0, 0.0
            ms_mode, ft_mode = "none", "none"

            # Domain 0 / Replay stream
            if self.use_replay or self.primary_task_id == 0:
                try:
                    batch_ms = next(ms_iter)  # CPU logical batch
                    lbl_ms = batch_ms.pop("label")
                    out_ms = self._stream_backward(
                        batch_cpu=batch_ms,
                        labels_cpu=lbl_ms,
                        criterion=self.criterion_ms,
                        loss_type=self.ms_loss_type,
                        loss_weight=self.ms_lambda,
                        microbatch_size=self.ms_cfg["micro_bs"],
                        scaler=scaler,
                        teacher=self.teacher if (self.dr_lambda > 0 and self.teacher is not None) else None,
                        dr_lambda=self.dr_lambda,
                        stream_name="MS",
                    )
                    l_ms_v = out_ms["loss"]
                    l_dr_v = out_ms["dr_loss"]
                    ms_mode = out_ms["mode"]
                except (StopIteration, AttributeError):
                    pass

            if self.empty_cache_between_streams and self.device_type == "cuda":
                torch.cuda.empty_cache()

            # Domain 1 / FT stream
            if self.use_replay or self.primary_task_id == 1:
                try:
                    batch_ft = next(ft_iter)  # CPU logical batch
                    lbl_ft = batch_ft.pop("label")
                    out_ft = self._stream_backward(
                        batch_cpu=batch_ft,
                        labels_cpu=lbl_ft,
                        criterion=self.criterion_ft,
                        loss_type=self.nonms_loss_type,
                        loss_weight=self.ft_lambda,
                        microbatch_size=self.ft_cfg["micro_bs"],
                        scaler=scaler,
                        teacher=None,
                        dr_lambda=0.0,
                        stream_name="FT",
                    )
                    l_ft_v = out_ft["loss"]
                    ft_mode = out_ft["mode"]
                except (StopIteration, AttributeError):
                    pass

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix({
                "MS": f"{l_ms_v:.6f}",
                "FT": f"{l_ft_v:.6f}",
                "DR": f"{l_dr_v:.6f}",
                "M": f"{ms_mode}/{ft_mode}",
            })

            metrics["ms_loss"].append(l_ms_v)
            metrics["ft_loss"].append(l_ft_v)
            metrics["dr_loss"].append(l_dr_v)
            metrics["ms_mode"].append(ms_mode)
            metrics["ft_mode"].append(ft_mode)

            if self.output_dir and (step + 1) % self.save_every_n_steps == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, f"pepcl_micro_step_{step + 1}.pt"),
                )

        return metrics
