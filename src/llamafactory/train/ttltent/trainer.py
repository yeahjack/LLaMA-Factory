# (文件头/导入保持不变)

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Seq2SeqTrainer
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)

# ===========================
# Loss Balancer Framework
# ===========================


class _BaseBalancer:
    """Base class for loss balancing strategies."""

    def __init__(self, finetuning_args: FinetuningArguments):
        self.args = finetuning_args

    def setup(self, trainer: TTLTENTTrainer) -> list[nn.Parameter]:
        return []

    def compute_weights(
        self,
        trainer: TTLTENTTrainer,
        base_w_ttl: float,
        base_w_tent: float,
        ttl_loss: torch.Tensor,
        tent_loss: torch.Tensor,
        context: dict[str, float],
    ):
        extra = torch.zeros((), device=ttl_loss.device, dtype=ttl_loss.dtype)
        return float(base_w_ttl), float(base_w_tent), extra


class _MovingAverageBalancer(_BaseBalancer):
    """EMA-normalize each loss, then renormalize to the same base sum."""

    def __init__(self, finetuning_args):
        super().__init__(finetuning_args)
        self.ema_ttl: torch.Tensor | None = None
        self.ema_tent: torch.Tensor | None = None
        self.momentum: float = float(getattr(finetuning_args, "bal_ema_momentum", 0.9))

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        device, dtype = ttl_loss.device, ttl_loss.dtype
        if self.ema_ttl is None:
            self.ema_ttl = ttl_loss.detach()
        else:
            self.ema_ttl = self.momentum * self.ema_ttl + (1 - self.momentum) * ttl_loss.detach()
        if self.ema_tent is None:
            self.ema_tent = tent_loss.detach()
        else:
            self.ema_tent = self.momentum * self.ema_tent + (1 - self.momentum) * tent_loss.detach()

        wt = base_w_ttl / (self.ema_ttl + 1e-8)
        we = base_w_tent / (self.ema_tent + 1e-8)
        base_sum = float(base_w_ttl + base_w_tent)
        s = (wt + we).item()
        scale = base_sum / (s + 1e-8)
        wt = float((wt * scale).item())
        we = float((we * scale).item())
        return wt, we, torch.zeros((), device=device, dtype=dtype)


class _DynamicWeightBalancer(_BaseBalancer):
    """DWA: use last two steps' loss ratio; temperature default=2.0."""

    def __init__(self, finetuning_args):
        super().__init__(finetuning_args)
        self.hist_ttl: list[float] = []
        self.hist_tent: list[float] = []
        self.temperature: float = float(getattr(finetuning_args, "dwa_temperature", 2.0))

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        if len(self.hist_ttl) < 2 or len(self.hist_tent) < 2:
            wt, we, extra = _BaseBalancer.compute_weights(
                self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context
            )
        else:
            r_ttl = self.hist_ttl[-1] / (self.hist_ttl[-2] + 1e-8)
            r_tent = self.hist_tent[-1] / (self.hist_tent[-2] + 1e-8)
            z_ttl = math.exp(r_ttl / self.temperature)
            z_tent = math.exp(r_tent / self.temperature)
            s = z_ttl + z_tent + 1e-8
            base_sum = float(base_w_ttl + base_w_tent)
            wt = float(base_sum * (z_ttl / s))
            we = float(base_sum * (z_tent / s))
            extra = torch.zeros_like(ttl_loss)
        self.hist_ttl.append(float(ttl_loss.detach().item()))
        self.hist_tent.append(float(tent_loss.detach().item()))
        if len(self.hist_ttl) > 1000:
            self.hist_ttl = self.hist_ttl[-500:]
            self.hist_tent = self.hist_tent[-500:]
        return wt, we, extra


class _GradientMagnitudeBalancer(_BaseBalancer):
    """GradNorm-lite: equalize gradient magnitudes on all trainable params."""

    def __init__(self, finetuning_args):
        super().__init__(finetuning_args)
        self.beta: float = float(getattr(finetuning_args, "gm_beta", 1.0))

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        params = [p for p in trainer.model.parameters() if p.requires_grad]
        if not params:
            return float(base_w_ttl), float(base_w_tent), torch.zeros_like(ttl_loss)

        g_t = torch.autograd.grad(ttl_loss, params, retain_graph=True, create_graph=False, allow_unused=True)
        g_e = torch.autograd.grad(tent_loss, params, retain_graph=True, create_graph=False, allow_unused=True)

        def _norm(gs):
            total = None
            for g in gs:
                if g is None:
                    continue
                v = (g.detach() * g.detach()).sum()
                total = v if total is None else total + v
            if total is None:
                return 0.0
            return float(torch.sqrt(total + 1e-12).item())

        ng_t = _norm(g_t)
        ng_e = _norm(g_e)
        if ng_t <= 0.0 or ng_e <= 0.0 or not math.isfinite(ng_t + ng_e):
            return float(base_w_ttl), float(base_w_tent), torch.zeros_like(ttl_loss)

        wt_raw = base_w_ttl / (ng_t**self.beta + 1e-8)
        we_raw = base_w_tent / (ng_e**self.beta + 1e-8)
        base_sum = float(base_w_ttl + base_w_tent)
        scale = base_sum / (wt_raw + we_raw + 1e-8)
        wt = float(wt_raw * scale)
        we = float(we_raw * scale)
        return wt, we, torch.zeros_like(ttl_loss)


class _AdaptiveScalingBalancer(_MovingAverageBalancer):
    """MA + information gating scaling."""

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        wt, we, extra = super().compute_weights(trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context)
        p_sel = max(1e-6, float(context.get("ttl_selected_ratio", 1.0)))
        valid_ratio = max(1e-6, float(context.get("tent_valid_ratio", 1.0)))
        wt *= math.sqrt(p_sel)
        we *= math.sqrt(valid_ratio)
        base_sum = float(base_w_ttl + base_w_tent)
        s = wt + we + 1e-8
        scale = base_sum / s
        wt *= scale
        we *= scale
        return wt, we, extra


class _UncertaintyBalancer(_BaseBalancer):
    """Homoscedastic uncertainty weighing with learnable log-sigmas."""

    def __init__(self, finetuning_args):
        super().__init__(finetuning_args)
        self.log_sigma_ttl = nn.Parameter(torch.zeros(()))
        self.log_sigma_tent = nn.Parameter(torch.zeros(()))

    def setup(self, trainer: TTLTENTTrainer) -> list[nn.Parameter]:
        return [self.log_sigma_ttl, self.log_sigma_tent]

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        sigma_t = torch.exp(self.log_sigma_ttl)
        sigma_e = torch.exp(self.log_sigma_tent)
        wt = float((base_w_ttl / (2.0 * (sigma_t**2) + 1e-8)).item())
        we = float((base_w_tent / (2.0 * (sigma_e**2) + 1e-8)).item())
        extra = self.log_sigma_ttl + self.log_sigma_tent
        # 交替时仅对活跃项计入 log_sigma
        if bool(getattr(trainer.finetuning_args, "alternating_training", False)):
            step_idx = (trainer.state.global_step or 0) + 1
            ttl_on = step_idx % 2 == 1
            tent_on = not ttl_on
            extra = (self.log_sigma_ttl if ttl_on else 0.0) + (self.log_sigma_tent if tent_on else 0.0)
        if not torch.is_tensor(extra):
            extra = torch.tensor(float(extra), device=ttl_loss.device, dtype=ttl_loss.dtype)
        return wt, we, extra


def _build_balancer(fa: FinetuningArguments) -> _BaseBalancer:
    name = str(getattr(fa, "loss_balancing_method", "static")).lower()
    if name == "moving_average":
        return _MovingAverageBalancer(fa)
    if name == "dynamic_weight":
        return _DynamicWeightBalancer(fa)
    if name == "gradient_magnitude":
        return _GradientMagnitudeBalancer(fa)
    if name == "adaptive_scaling":
        return _AdaptiveScalingBalancer(fa)
    if name == "uncertainty":
        return _UncertaintyBalancer(fa)
    return _BaseBalancer(fa)


class TTLTENTTrainer(Seq2SeqTrainer):
    """Combined TTL (prompt NLL) + TENT (continuation entropy) with single-pass reuse."""

    def __init__(
        self,
        finetuning_args: FinetuningArguments,
        *args,
        **kwargs,
    ):
        self.precomputed_predictions: list[list[int]] | None = kwargs.pop("precomputed_predictions", None)
        self.gen_model_mode: str = kwargs.pop("gen_model_mode", getattr(finetuning_args, "gen_model", "simultaneous"))

        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args

        self.ref_sentence_ce: dict[int, float] | None = None
        self._precompute_ptr: int = 0
        self.token_log: list[dict] = []

        if hasattr(self, "processing_class") and self.processing_class is not None:
            try:
                self.processing_class.padding_side = "left"
                logger.info_rank0("Set tokenizer padding_side='left' for generation compatibility.")
            except Exception as e:
                logger.warning_rank0(f"Could not set padding_side: {e}")

        self.teacher: nn.Module | None = None
        if bool(getattr(self.finetuning_args, "use_kl_regularization", False)):
            try:
                self.teacher = copy.deepcopy(self.model)
                for p in self.teacher.parameters():
                    p.requires_grad_(False)
                self.teacher.eval()
                logger.info_rank0("[TTLTENT] Teacher model created for KL regularization.")
            except Exception as e:
                logger.warning_rank0(f"Failed to copy teacher model for KL: {e}")
                self.teacher = None

        self._balancer: _BaseBalancer = _build_balancer(self.finetuning_args)
        self._extra_balancer_params: list[nn.Parameter] = []

    # Optimizer / Scheduler / Sampler 与上版一致，略

    def _get_pad_eos_ids(self) -> tuple[int, int]:
        # 与上版一致
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        eos_id = getattr(self.processing_class, "eos_token_id", None)
        if pad_id is None and hasattr(self, "tokenizer") and self.tokenizer is not None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if eos_id is None and hasattr(self, "tokenizer") and self.tokenizer is not None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        if eos_id is None:
            eos_id = 0
        return int(pad_id), int(eos_id)

    def _pad_and_stack(self, sequences: list[list[int]], pad_id: int, device, dtype) -> torch.Tensor:
        # 与上版一致
        bsz = len(sequences)
        max_len = max((len(s) for s in sequences), default=0)
        out = torch.full((bsz, max_len), pad_id, dtype=dtype, device=device)
        for i, s in enumerate(sequences):
            if len(s) > 0:
                out[i, : len(s)] = torch.tensor(s, dtype=dtype, device=device)
        return out

    def _fetch_precomputed_continuations(
        self, bsz: int, limit: int | None, pad_id: int, device, dtype
    ) -> torch.Tensor:
        # 与上版一致
        cont_list: list[list[int]] = []
        for _ in range(bsz):
            if self.precomputed_predictions is not None and self._precompute_ptr < len(self.precomputed_predictions):
                seq = self.precomputed_predictions[self._precompute_ptr]
                self._precompute_ptr += 1
            else:
                seq = []
            if limit is not None and limit > 0:
                seq = seq[:limit]
            cont_list.append(seq)
        return self._pad_and_stack(cont_list, pad_id=pad_id, device=device, dtype=dtype)

    @override
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # 0) 清理 labels
        if "labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}

        input_ids: torch.Tensor = inputs["input_ids"]  # [B, P]
        attn: torch.Tensor | None = inputs.get("attention_mask", None)  # [B, P] or None
        device = input_ids.device
        pad_id, eos_id = self._get_pad_eos_ids()

        fa = self.finetuning_args
        generation_len = int(getattr(fa, "generation_len", 0))
        prompt_len_for_ttl = input_ids.size(1)  # P

        # 1) 生成 continuation 并拼接
        if generation_len != 0:
            was_train = self.model.training
            self.model.eval()
            with torch.no_grad():
                if self.gen_model_mode == "precompute" and self.precomputed_predictions is not None:
                    limit = None if generation_len == -1 else max(0, generation_len)
                    cont = self._fetch_precomputed_continuations(
                        bsz=input_ids.size(0),
                        limit=limit,
                        pad_id=pad_id,
                        device=device,
                        dtype=input_ids.dtype,
                    )  # [B, Lc]
                    generated_tokens = torch.cat([input_ids, cont], dim=1)  # [B, P+Lc]
                else:
                    max_tokens = 2048 if generation_len == -1 else max(0, generation_len)
                    generated_tokens = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=max_tokens,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                        do_sample=False,
                        top_k=None,
                        top_p=None,
                        temperature=1.0,
                    )  # [B, P+Lc*]
            if was_train:
                self.model.train()
            tent_start = prompt_len_for_ttl
        else:
            # 与 TENT/EATA 对齐：无生成段时对整段输入做熵最小化
            generated_tokens = input_ids  # [B, P]==[B, T]
            tent_start = 0

        # 统一的对齐起点（关键修正）
        # [FIX] tok_start 对齐：有生成 -> tok_start=P；无生成 -> tok_start=1
        tok_start = tent_start if tent_start > 0 else 1

        # 2) 一次前向
        outputs: CausalLMOutput = self.model(input_ids=generated_tokens)
        logits = outputs.logits  # [B, T, V]
        B, T, V = logits.size()

        # 3) TTL：基于 prompt 段的句级 CE（NLL）
        if prompt_len_for_ttl > 1:
            ttl_logits = logits[:, : prompt_len_for_ttl - 1, :]  # [B, P-1, V]
            ttl_labels = generated_tokens[:, 1:prompt_len_for_ttl]  # [B, P-1]
            criterion = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
            ttl_token_nll = criterion(ttl_logits.reshape(-1, V), ttl_labels.reshape(-1)).view(B, -1)  # [B, P-1]
            if attn is not None:
                ttl_mask = (attn[:, 1:prompt_len_for_ttl] != 0).to(ttl_token_nll.dtype)  # [B, P-1]
            else:
                ttl_mask = torch.ones_like(ttl_token_nll, dtype=ttl_token_nll.dtype)
            ttl_denom = ttl_mask.sum(dim=1).clamp_min(1.0)
            sentence_ce_train = (ttl_token_nll * ttl_mask).sum(dim=1) / ttl_denom  # [B]
        else:
            sentence_ce_train = torch.zeros((B,), device=device, dtype=logits.dtype)
            ttl_token_nll = torch.zeros((B, 0), device=device, dtype=logits.dtype)
            ttl_mask = torch.zeros((B, 0), device=device, dtype=logits.dtype)

        # 4) TENT：基于 continuation（或整段）计算 token 熵
        if tok_start <= T - 1:
            gen_logits = logits[:, tok_start - 1 : T - 1, :]  # [B, Lg, V]，Lg = T - tok_start
            gen_tokens = generated_tokens[:, tok_start:]  # [B, Lg]
            log_probs = F.log_softmax(gen_logits, dim=-1)
            probs = log_probs.exp()
            entropy_tok = -(probs * log_probs).sum(dim=-1)  # [B, Lg]
            ent_mask = gen_tokens != pad_id
            if bool(getattr(fa, "ignore_eos_in_entropy", True)):
                ent_mask = ent_mask & (gen_tokens != eos_id)
            ent_mask = ent_mask.to(entropy_tok.dtype)  # [B, Lg]
            tent_seq_entropy = (entropy_tok * ent_mask).sum(dim=1) / ent_mask.sum(dim=1).clamp_min(1e-8)  # [B]
        else:
            entropy_tok = torch.zeros((B, 0), device=device, dtype=logits.dtype)
            ent_mask = torch.zeros((B, 0), device=device, dtype=logits.dtype)
            tent_seq_entropy = torch.zeros((B,), device=device, dtype=logits.dtype)

        # 5) 参考 CE（用于 gating）
        ref_mode = str(getattr(fa, "ttl_ref_mode", "precompute")).lower()
        if ref_mode not in {"precompute", "simultaneous"}:
            raise ValueError(f"Unsupported ttl_ref_mode: {ref_mode}")
        if ref_mode == "simultaneous":
            ce_ref = sentence_ce_train.detach()
        else:
            if "example_id" not in inputs:
                raise RuntimeError("ttl_ref_mode=precompute 需要 batch 中包含 example_id。")
            if self.ref_sentence_ce is None:
                raise RuntimeError("参考 CE 尚未预计算，请先设置 trainer.ref_sentence_ce。")
            ex_ids = inputs["example_id"]
            if isinstance(ex_ids, torch.Tensor):
                ex_ids = ex_ids.tolist()
            ce_ref = torch.tensor([self.ref_sentence_ce[int(e)] for e in ex_ids], dtype=logits.dtype, device=device)

        # 6) TTL gating & loss
        ttl_gating: str = str(getattr(fa, "ttl_gating", "ttl")).lower()
        threshold = float(getattr(fa, "ttl_threshold", 3.0))
        scaler = float(getattr(fa, "ttl_sample_efficiency_scaler", 0.1))
        apply_ttl_gate = ttl_gating in {"ttl", "all"}
        apply_tent_gate = ttl_gating in {"tent", "all"}

        if apply_ttl_gate:
            delta = (ce_ref - threshold).clamp_max(20.0)
            mask_s = (ce_ref > threshold).to(logits.dtype)
            coeff = scaler * torch.exp(delta.detach())
            ttl_vec = sentence_ce_train * coeff * mask_s
            ttl_loss = (ttl_vec.sum() / mask_s.sum().clamp_min(1.0)) if mask_s.sum() != 0 else sentence_ce_train.mean()
        else:
            ttl_loss = sentence_ce_train.mean()

        # 7) TENT gating & loss（样本级 gating，无加权）
        if apply_tent_gate:
            mask_s = (ce_ref > threshold).to(logits.dtype)
            tent_vec = tent_seq_entropy * mask_s
            tent_loss = (
                (tent_vec.sum() / mask_s.sum().clamp_min(1.0)) if mask_s.sum() != 0 else tent_seq_entropy.mean()
            )
        else:
            tent_loss = tent_seq_entropy.mean()

        # 8) Alternating（A 规则：奇步 TTL / 偶步 TENT）
        alternating = bool(getattr(fa, "alternating_training", False))
        ttl_active = True
        tent_active = True
        if alternating:
            step_idx = (self.state.global_step or 0) + 1
            ttl_active = step_idx % 2 == 1
            tent_active = not ttl_active

        # 9) Loss Balancing
        base_w_ttl = float(getattr(fa, "loss_weight_ttl", 1.0))
        base_w_tent = float(getattr(fa, "loss_weight_tent", 1.0))
        ttl_selected_ratio = (
            float((ce_ref > threshold).float().mean().item()) if (apply_ttl_gate or apply_tent_gate) else 1.0
        )
        tent_valid_ratio = float(ent_mask.mean().item()) if ent_mask.numel() > 0 else 1.0
        context = {"ttl_selected_ratio": ttl_selected_ratio, "tent_valid_ratio": tent_valid_ratio}

        extra_loss = torch.zeros((), device=device, dtype=logits.dtype)
        if alternating:
            w_ttl = base_w_ttl if ttl_active else 0.0
            w_tent = base_w_tent if tent_active else 0.0
            from math import isfinite  # noqa

            if isinstance(self._balancer, _UncertaintyBalancer):
                _wt, _we, extra_loss = self._balancer.compute_weights(self, 1.0, 1.0, ttl_loss, tent_loss, context)
        else:
            w_ttl, w_tent, extra_loss = self._balancer.compute_weights(
                self, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context
            )

        # 10) KL 正则（始终加，但仅由 kl_weight 调节强度）
        kl_loss = torch.zeros((), device=device, dtype=logits.dtype)
        if bool(getattr(fa, "use_kl_regularization", False)) and self.teacher is not None:
            try:
                with torch.no_grad():
                    t_outputs: CausalLMOutput = self.teacher(input_ids=generated_tokens)
                    t_logits = t_outputs.logits
                cur_lp = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
                tch_lp = F.log_softmax(t_logits[:, :-1, :], dim=-1)  # [B, T-1, V]
                cur_p = cur_lp.exp()
                targets = generated_tokens[:, 1:]  # [B, T-1]
                kl_mask = targets != pad_id
                if bool(getattr(fa, "ignore_eos_in_entropy", True)):
                    kl_mask = kl_mask & (targets != eos_id)
                kl_mask = kl_mask.to(cur_lp.dtype)  # [B, T-1]
                kl_tok = (cur_p * (cur_lp - tch_lp)).sum(dim=-1)  # [B, T-1]
                denom = kl_mask.sum().clamp_min(1.0)
                kl_loss = (kl_tok * kl_mask).sum() / denom
            except Exception as e:
                logger.warning_rank0(f"[TTLTENT] KL regularization failed on this batch: {e}")
                kl_loss = torch.zeros((), device=device, dtype=logits.dtype)

        kl_weight = float(getattr(fa, "kl_weight", 0.0)) if bool(getattr(fa, "use_kl_regularization", False)) else 0.0

        # 11) Total loss
        ttl_term = (ttl_loss * w_ttl) if ttl_active else torch.zeros((), device=device, dtype=logits.dtype)
        tent_term = (tent_loss * w_tent) if tent_active else torch.zeros((), device=device, dtype=logits.dtype)
        total_loss = ttl_term + tent_term + kl_weight * kl_loss + extra_loss

        # 12) token_log（保持与切片一致，使用 tok_start）
        try:
            with torch.no_grad():
                ex_ids = inputs.get("example_id", None)
                if isinstance(ex_ids, torch.Tensor):
                    ex_ids = ex_ids.detach().cpu().tolist()
                else:
                    ex_ids = [-1] * B

                prompt_ids_list, gen_ids_list = [], []
                prompt_nll_list, gen_entropy_list = [], []

                for i in range(B):
                    # prompt tokens（按 attention 截断）
                    if attn is not None:
                        plen = int(attn[i].sum().item())
                    else:
                        plen = int(prompt_len_for_ttl)
                    prompt_ids = input_ids[i, :plen].detach().cpu().tolist()
                    prompt_ids_list.append(prompt_ids)

                    # generation tokens（使用 tok_start 对齐）
                    if T >= tok_start + 1:
                        gen_ids = generated_tokens[i, tok_start:].detach().cpu()
                        gen_mask_i = (
                            ent_mask[i].bool().detach().cpu()
                            if ent_mask.numel() > 0
                            else torch.tensor([], dtype=torch.bool)
                        )
                        gen_ids = (
                            gen_ids[gen_mask_i]
                            if gen_mask_i.numel() > 0
                            else torch.tensor([], dtype=generated_tokens.dtype)
                        )
                        gen_ids_list.append(gen_ids.tolist())
                    else:
                        gen_ids_list.append([])

                    # prompt token-level NLL
                    if ttl_token_nll.size(1) > 0:
                        mask_i = ttl_mask[i].bool().detach().cpu()
                        nll_i = ttl_token_nll[i].detach().cpu()
                        nll_i = nll_i[mask_i] if mask_i.numel() > 0 else torch.tensor([])
                        prompt_nll_list.append([float(x) for x in nll_i.tolist()])
                    else:
                        prompt_nll_list.append([])

                    # generation token-level entropy
                    if entropy_tok.size(1) > 0:
                        e_mask_i = ent_mask[i].bool().detach().cpu()
                        ent_i = entropy_tok[i].detach().cpu()
                        ent_i = ent_i[e_mask_i] if e_mask_i.numel() > 0 else torch.tensor([])
                        gen_entropy_list.append([float(x) for x in ent_i.tolist()])
                    else:
                        gen_entropy_list.append([])

                for i in range(B):
                    self.token_log.append(
                        {
                            "example_id": int(ex_ids[i]) if ex_ids else -1,
                            "prompt_token_ids": prompt_ids_list[i],
                            "generated_token_ids": gen_ids_list[i],
                            "prompt_token_nll": prompt_nll_list[i],
                            "generation_token_entropy": gen_entropy_list[i],
                        }
                    )
        except Exception as e:
            logger.warning_rank0(f"[TTLTENT] token_log append failed: {e}")

        return (total_loss, outputs) if return_outputs else total_loss
