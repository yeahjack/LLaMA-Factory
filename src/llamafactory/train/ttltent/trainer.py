# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
from contextlib import nullcontext as _nullctx
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
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


# ============================================================
# Loss Balancer Framework
# ============================================================


class _BaseBalancer(nn.Module):
    """Base class for 2-branch (TTL, TENT) loss balancing."""

    def __init__(self):
        super().__init__()

    def compute_weights(
        self,
        trainer: "TTLTENTTrainer",
        base_w_ttl: float,
        base_w_tent: float,
        ttl_loss: torch.Tensor,
        tent_loss: torch.Tensor,
        context: dict,
    ) -> tuple[float, float, torch.Tensor]:
        # default: static weights, no extra loss
        return float(base_w_ttl), float(base_w_tent), ttl_loss.new_zeros(())


class _StaticBalancer(_BaseBalancer):
    pass


class _MovingAverageBalancer(_BaseBalancer):
    """EMA of per-branch loss; inverse-proportional weighting to equalize."""

    def __init__(self, momentum: float = 0.9):
        super().__init__()
        self.register_buffer("ema_ttl", torch.tensor(1.0))
        self.register_buffer("ema_tent", torch.tensor(1.0))
        self.momentum = momentum

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        with torch.no_grad():
            self.ema_ttl.mul_(self.momentum).add_(float(ttl_loss.detach().clamp_min(1e-12)) * (1 - self.momentum))
            self.ema_tent.mul_(self.momentum).add_(float(tent_loss.detach().clamp_min(1e-12)) * (1 - self.momentum))
            inv_ttl = 1.0 / float(self.ema_ttl)
            inv_tent = 1.0 / float(self.ema_tent)
            s = inv_ttl + inv_tent + 1e-12
            w_ttl = base_w_ttl * inv_ttl / s * 2.0
            w_tent = base_w_tent * inv_tent / s * 2.0
        return w_ttl, w_tent, ttl_loss.new_zeros(())


class _DynamicWeightBalancer(_BaseBalancer):
    """Simple loss-ratio balancing."""

    def __init__(self, beta: float = 0.9):
        super().__init__()
        self.register_buffer("ema_total", torch.tensor(1.0))
        self.beta = beta

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        with torch.no_grad():
            tot = float(ttl_loss.detach() + tent_loss.detach() + 1e-12)
            self.ema_total.mul_(self.beta).add_(tot * (1 - self.beta))
            r_ttl = float(ttl_loss.detach()) / float(self.ema_total + 1e-12)
            r_tent = float(tent_loss.detach()) / float(self.ema_total + 1e-12)
            s = r_ttl + r_tent + 1e-12
            w_ttl = base_w_ttl * r_ttl / s * 2.0
            w_tent = base_w_tent * r_tent / s * 2.0
        return w_ttl, w_tent, ttl_loss.new_zeros(())


class _GradientMagnitudeBalancer(_BaseBalancer):
    """Balance by matching gradient magnitudes (approx.)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        # Approximate gradient norms by loss values (proxy) to avoid 2x autograd
        L_ttl = float(ttl_loss.detach().abs().clamp_min(self.eps))
        L_tent = float(tent_loss.detach().abs().clamp_min(self.eps))
        inv_ttl = 1.0 / L_ttl
        inv_tent = 1.0 / L_tent
        s = inv_ttl + inv_tent
        w_ttl = base_w_ttl * inv_ttl / s * 2.0
        w_tent = base_w_tent * inv_tent / s * 2.0
        return w_ttl, w_tent, ttl_loss.new_zeros(())


class _AdaptiveScalingBalancer(_BaseBalancer):
    """Keep weights within a floor/ceil window; lean to equalization."""

    def __init__(self, floor: float = 1e-3, ceil: float = 1e3):
        super().__init__()
        self.floor = floor
        self.ceil = ceil

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        L_ttl = float(ttl_loss.detach().clamp_min(1e-12))
        L_tent = float(tent_loss.detach().clamp_min(1e-12))
        r = L_tent / L_ttl
        r = max(self.floor, min(self.ceil, r))
        w_ttl = base_w_ttl
        w_tent = base_w_tent * r
        return w_ttl, w_tent, ttl_loss.new_zeros(())


class _UncertaintyBalancer(_BaseBalancer):
    """Kendall et al. (multi-task): w_i = 1/(2 sigma_i^2); extra loss = log sigma_i."""

    def __init__(self, trainer: TTLTENTTrainer):
        super().__init__()
        self.log_var_ttl = nn.Parameter(torch.zeros(()))
        self.log_var_tent = nn.Parameter(torch.zeros(()))
        trainer._extra_balancer_params.extend([self.log_var_ttl, self.log_var_tent])

    def compute_weights(self, trainer, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context):
        w_ttl = base_w_ttl * torch.exp(-self.log_var_ttl).item() / 2.0
        w_tent = base_w_tent * torch.exp(-self.log_var_tent).item() / 2.0
        extra = self.log_var_ttl + self.log_var_tent
        return w_ttl, w_tent, extra


def _build_balancer(fa: FinetuningArguments) -> _BaseBalancer | None:
    name = str(getattr(fa, "loss_balancing_method", "static")).lower()
    if name in ("static", "st"):
        return _StaticBalancer()
    if name in ("moving_average", "ma"):
        m = float(getattr(fa, "ma_momentum", 0.9))
        return _MovingAverageBalancer(momentum=m)
    if name in ("dynamic_weight", "dw"):
        b = float(getattr(fa, "dw_beta", 0.9))
        return _DynamicWeightBalancer(beta=b)
    if name in ("gradient_magnitude", "gm"):
        e = float(getattr(fa, "gm_eps", 1e-8))
        return _GradientMagnitudeBalancer(eps=e)
    if name in ("adaptive_scaling", "as"):
        f = float(getattr(fa, "as_floor", 1e-3))
        c = float(getattr(fa, "as_ceil", 1e3))
        return _AdaptiveScalingBalancer(floor=f, ceil=c)
    if name in ("uncertainty", "uc"):
        return None  # 占位：由 Trainer 在 __init__ 内实例化（需要注册参数）
    logger.warning_rank0(f"Unknown loss_balancing_method={name}, fallback to static.")
    return _StaticBalancer()


# ============================================================
# TTLTENT Trainer
# ============================================================


class TTLTENTTrainer(Seq2SeqTrainer):
    """Combined TTL (prompt NLL) + TENT (continuation entropy, supports EM-FT) + reverse-KL(student‖teacher)."""

    def __init__(
        self,
        finetuning_args: FinetuningArguments,
        *args,
        **kwargs,
    ):
        # TENT: precomputed continuations
        self.precomputed_predictions: list[list[int]] | None = kwargs.pop("precomputed_predictions", None)
        self.gen_model_mode: str = kwargs.pop("gen_model_mode", getattr(finetuning_args, "gen_model", "simultaneous"))

        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args

        # TTL ref CE (only if ttl_ref_mode="precompute")
        self.ref_sentence_ce: dict[int, float] | None = None
        self._precompute_ptr: int = 0

        # token-level diagnostic log
        self.token_log: list[dict] = []

        # Padding side hint
        if hasattr(self, "processing_class") and self.processing_class is not None:
            try:
                self.processing_class.padding_side = "left"
                logger.info_rank0("Set tokenizer padding_side='left' for generation compatibility.")
            except Exception as e:
                logger.warning_rank0(f"Could not set padding_side: {e}")

        # === 教师构造策略：仅在 use_kl_regularization 为 True 且存在续写时才尝试构造 ===
        if bool(getattr(self.finetuning_args, "use_kl_regularization", False)) and int(
            getattr(self.finetuning_args, "generation_len", 0)
        ) != 0:
            try:
                unwrapped = getattr(self.accelerator, "unwrap_model", lambda x: x)(self.model)
                has_disable = getattr(unwrapped, "disable_adapter", None) is not None
            except Exception:
                unwrapped = self.model
                has_disable = False

            if has_disable:
                # 使用 LoRA 的 disable_adapter() 作为教师路径；不创建 teacher 副本
                self.teacher: nn.Module | None = None
                logger.info_rank0("[TTLTENT] Using disable_adapter() for teacher path. No deepcopy allocated.")
            else:
                # 回退：深拷贝一份冻结教师
                try:
                    self.teacher = copy.deepcopy(self.model)
                    for p in self.teacher.parameters():
                        p.requires_grad_(False)
                    self.teacher.eval()
                    logger.info_rank0("[TTLTENT] Teacher model created via deepcopy (fallback).")
                except Exception as e:
                    logger.warning_rank0(f"Failed to copy teacher model (KL may be disabled): {e}")
                    self.teacher = None
        else:
            # KL 关闭或无续写：不构造 teacher，不使用 disable_adapter 路径
            self.teacher = None
            logger.info_rank0("[TTLTENT] KL disabled or no generation: no teacher/disable_adapter will be used.")

        # Balancer
        base_balancer = _build_balancer(self.finetuning_args)
        self._extra_balancer_params: list[nn.Parameter] = []
        if base_balancer is None:
            self._balancer = _UncertaintyBalancer(self)
        else:
            self._balancer = base_balancer

        # Step-wise history for plotting
        self._hist_steps: list[int] = []
        self._hist_ttl: list[float] = []
        self._hist_tent: list[float] = []
        self._hist_wttl: list[float] = []
        self._hist_wtent: list[float] = []

        # Micro-step accumulators
        self._acc_step: int | None = None
        self._acc_count: int = 0
        self._acc_ttl_sum: float = 0.0
        self._acc_tent_sum: float = 0.0
        self._acc_wttl_sum: float = 0.0
        self._acc_wtent_sum: float = 0.0

    # ------------------------ Optimizer / Scheduler / Sampler ------------------------ #
    @override
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
            # attach extra balancer params if any
            if self._extra_balancer_params:
                group = {"params": self._extra_balancer_params, "lr": float(getattr(self.args, "learning_rate", 5e-5))}
                try:
                    self.optimizer.add_param_group(group)
                    logger.info_rank0(f"Added {len(self._extra_balancer_params)} balancer params to optimizer.")
                except Exception as e:
                    logger.warning_rank0(f"Failed to add balancer params to optimizer: {e}")
        return super().create_optimizer()

    @override
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs):
        if getattr(self.finetuning_args, "disable_shuffling", False):
            import torch as _torch

            return _torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    # ------------------------ Utilities ------------------------ #
    def _get_pad_eos_ids(self) -> tuple[int, int]:
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

    def _concat_prompt_with_continuations(
        self,
        input_ids: torch.Tensor,  # [B, Pmax]
        attn: torch.Tensor | None,  # [B, Pmax] or None
        cont: torch.Tensor,  # [B, Cmax] (right-padded)
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将每个样本的真实 prompt（去除右侧 PAD）与该样本的续写拼接，返回：
          - generated_tokens: [B, Tmax]
          - attention_mask_gen: [B, Tmax]
          - prompt_lens: [B]（每条样本真实 prompt 长度）
        """
        device = input_ids.device
        B = input_ids.size(0)
        if attn is not None:
            prompt_lens = attn.sum(dim=1).to(torch.long)  # [B]
        else:
            prompt_lens = (input_ids != pad_id).sum(dim=1).to(torch.long)

        cont_lens = (cont != pad_id).sum(dim=1).to(torch.long)  # [B]

        rows: list[list[int]] = []
        max_len = 0
        for i in range(B):
            plen = int(prompt_lens[i].item())
            clen = int(cont_lens[i].item())
            p_tokens = input_ids[i, :plen].tolist()
            c_tokens = cont[i, :clen].tolist() if clen > 0 else []
            row = p_tokens + c_tokens
            rows.append(row)
            if len(row) > max_len:
                max_len = len(row)

        gen_tokens = torch.full((B, max_len), pad_id, dtype=input_ids.dtype, device=device)
        attn_gen = torch.zeros((B, max_len), dtype=input_ids.dtype, device=device)
        for i, row in enumerate(rows):
            if len(row) > 0:
                gen_tokens[i, : len(row)] = torch.tensor(row, dtype=input_ids.dtype, device=device)
                attn_gen[i, : len(row)] = 1

        return gen_tokens, attn_gen, prompt_lens

    # ------------------------ Step-wise history helpers ------------------------ #
    def _accumulate_for_step(self, ttl_loss_val: float, tent_loss_val: float, w_ttl_raw: float, w_tent_raw: float):
        cur = int(self.state.global_step or 0)
        if self._acc_step is None:
            self._acc_step = cur
        if cur != self._acc_step:
            # finalize previous
            if self._acc_count > 0:
                self._hist_steps.append(self._acc_step)
                self._hist_ttl.append(self._acc_ttl_sum / self._acc_count)
                self._hist_tent.append(self._acc_tent_sum / self._acc_count)
                self._hist_wttl.append(self._acc_wttl_sum / self._acc_count)
                self._hist_wtent.append(self._acc_wtent_sum / self._acc_count)
            # reset for new step
            self._acc_step = cur
            self._acc_count = 0
            self._acc_ttl_sum = 0.0
            self._acc_tent_sum = 0.0
            self._acc_wttl_sum = 0.0
            self._acc_wtent_sum = 0.0
        self._acc_count += 1
        self._acc_ttl_sum += float(ttl_loss_val)
        self._acc_tent_sum += float(tent_loss_val)
        self._acc_wttl_sum += float(w_ttl_raw)
        self._acc_wtent_sum += float(w_tent_raw)

    def _finalize_history(self):
        if self._acc_step is not None and self._acc_count > 0:
            self._hist_steps.append(self._acc_step)
            self._hist_ttl.append(self._acc_ttl_sum / self._acc_count)
            self._hist_tent.append(self._acc_tent_sum / self._acc_count)
            self._hist_wttl.append(self._acc_wttl_sum / self._acc_count)
            self._hist_wtent.append(self._acc_wtent_sum / self._acc_count)
            # reset
            self._acc_step = None
            self._acc_count = 0
            self._acc_ttl_sum = 0.0
            self._acc_tent_sum = 0.0
            self._acc_wttl_sum = 0.0
            self._acc_wtent_sum = 0.0

    def export_diagnostics_plots(self, out_dir: str):
        """Save ttl_loss.png, tent_entropy_loss.png, balancer_weights.png into out_dir."""
        try:
            import os

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # flush pending step
            self._finalize_history()

            if len(self._hist_steps) == 0:
                logger.warning_rank0("[TTLTENT] No history to plot yet.")
                return

            os.makedirs(out_dir, exist_ok=True)

            # TTL loss
            plt.figure()
            plt.plot(self._hist_steps, self._hist_ttl, label="TTL (CE on prompt)")
            plt.xlabel("step")
            plt.ylabel("ttl_loss")
            plt.title("TTL loss vs step")
            plt.legend()
            ttl_path = os.path.join(out_dir, "ttl_loss.png")
            plt.savefig(ttl_path, bbox_inches="tight")
            plt.close()

            # TENT entropy loss
            plt.figure()
            plt.plot(self._hist_steps, self._hist_tent, label="TENT (entropy on continuation)")
            plt.xlabel("step")
            plt.ylabel("tent_entropy_loss")
            plt.title("TENT entropy loss vs step")
            plt.legend()
            tent_path = os.path.join(out_dir, "tent_entropy.png")
            plt.savefig(tent_path, bbox_inches="tight")
            plt.close()

            # Balancer weights
            plt.figure()
            plt.plot(self._hist_steps, self._hist_wttl, label="w_ttl (balancer)")
            plt.plot(self._hist_steps, self._hist_wtent, label="w_tent (balancer)")
            plt.xlabel("step")
            plt.ylabel("weight")
            plt.title("Balancer weights vs step")
            plt.legend()
            w_path = os.path.join(out_dir, "balancer_weights.png")
            plt.savefig(w_path, bbox_inches="tight")
            plt.close()

            logger.info_rank0(f"[TTLTENT] Saved plots: {ttl_path}, {tent_path}, {w_path}")
        except Exception as e:
            logger.warning_rank0(f"[TTLTENT] Failed to export plots: {e}")

    # ------------------------ Core compute_loss ------------------------ #
    @override
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # Remove supervised labels (test-time setting)
        if "labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}

        input_ids: torch.Tensor = inputs["input_ids"]  # [B, P]
        attn: torch.Tensor | None = inputs.get("attention_mask", None)  # [B, P] or None
        device = input_ids.device
        pad_id, eos_id = self._get_pad_eos_ids()
        fa = self.finetuning_args
        generation_len = int(getattr(fa, "generation_len", 0))
        prompt_max_len = input_ids.size(1)  # Pmax

        # 1) Generate continuation if needed
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
                    )  # [B, Cmax]

                    # 将续写拼到每条样本真实 prompt 末尾（无 PAD 间隙），并构造 attention_mask_gen
                    generated_tokens, attention_mask_gen, prompt_lens = self._concat_prompt_with_continuations(
                        input_ids=input_ids,
                        attn=attn,
                        cont=cont,
                        pad_id=pad_id,
                    )  # [B, T], [B, T], [B]
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
                    )  # [B, T]
                    # 续写后 attention_mask（不改变数值路径）
                    attention_mask_gen = (generated_tokens != pad_id).to(input_ids.dtype)
                    # 每条样本真实 prompt 长度
                    if attn is not None:
                        prompt_lens = attn.sum(dim=1).to(torch.long)
                    else:
                        prompt_lens = (input_ids != pad_id).sum(dim=1).to(torch.long)

            if was_train:
                self.model.train()
        else:
            # No generation: still do TENT on full input, but KL=0 by spec
            generated_tokens = input_ids
            attention_mask_gen = attn if attn is not None else (generated_tokens != pad_id).to(input_ids.dtype)
            # 每条样本真实 prompt 长度
            if attn is not None:
                prompt_lens = attn.sum(dim=1).to(torch.long)
            else:
                prompt_lens = (input_ids != pad_id).sum(dim=1).to(torch.long)

        # 2) One forward on (prompt+continuation) with explicit attention mask
        outputs: CausalLMOutput = self.model(input_ids=generated_tokens, attention_mask=attention_mask_gen)
        logits = outputs.logits  # [B, T, V]
        B, T, V = logits.size()

        # 3) TTL: sentence-level CE (prompt only, masked by original attn)
        if prompt_max_len > 1:
            ttl_logits = logits[:, : prompt_max_len - 1, :]  # [B, Pmax-1, V]
            ttl_labels = generated_tokens[:, 1:prompt_max_len]  # [B, Pmax-1]
            criterion = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
            ttl_token_nll = criterion(ttl_logits.reshape(-1, V), ttl_labels.reshape(-1)).view(B, -1)  # [B, Pmax-1]
            if attn is not None:
                ttl_mask = (attn[:, 1:prompt_max_len] != 0).to(ttl_token_nll.dtype)  # [B, Pmax-1]
            else:
                ttl_mask = torch.ones_like(ttl_token_nll, dtype=ttl_token_nll.dtype)
            ttl_denom = ttl_mask.sum(dim=1).clamp_min(1.0)
            sentence_ce_train = (ttl_token_nll * ttl_mask).sum(dim=1) / ttl_denom  # [B]
        else:
            sentence_ce_train = torch.zeros((B,), device=device, dtype=logits.dtype)
            ttl_token_nll = torch.zeros((B, 0), device=device, dtype=logits.dtype)
            ttl_mask = torch.zeros((B, 0), device=device, dtype=logits.dtype)

        # 4) TENT: entropy on continuation (or full input if no generation)
        # 统一在全时刻上计算 log_probs / entropy，再用 mask 选择续写段（更易与可变 prompt 对齐）
        if T >= 2:
            all_logits_next = logits[:, :-1, :]  # [B, T-1, V] -> 对应预测 generated_tokens[:, 1:]
            all_tokens_next = generated_tokens[:, 1:]  # [B, T-1]
            log_probs_all = F.log_softmax(all_logits_next, dim=-1)  # [B, T-1, V]
            probs_all = log_probs_all.exp()
            entropy_tok = -(probs_all * log_probs_all).sum(dim=-1)  # [B, T-1]

            # 续写段掩码：位置 j 对应预测 token 索引 (j+1)，续写从索引 >= prompt_lens[i]
            idx = torch.arange(T - 1, device=device).unsqueeze(0).expand(B, -1)  # [B, T-1]
            cont_mask = idx >= prompt_lens.unsqueeze(1)  # [B, T-1] (bool)
            tok_mask = (all_tokens_next != pad_id) & (all_tokens_next != eos_id)  # [B, T-1] (bool)
            final_mask = (cont_mask & tok_mask).to(entropy_tok.dtype)  # [B, T-1]

            # mean entropy per sequence (original TENT)
            tent_seq_entropy = (entropy_tok * final_mask).sum(dim=1) / final_mask.sum(dim=1).clamp_min(1e-8)  # [B]
            # EM-FT: path-total entropy per sequence (sum over tokens)
            tent_seq_total_entropy = (entropy_tok * final_mask).sum(dim=1)  # [B]
        else:
            log_probs_all = torch.zeros((B, 0, V), device=device, dtype=logits.dtype)
            entropy_tok = torch.zeros((B, 0), device=device, dtype=logits.dtype)
            final_mask = torch.zeros((B, 0), device=device, dtype=logits.dtype)
            tent_seq_entropy = torch.zeros((B,), device=device, dtype=logits.dtype)
            tent_seq_total_entropy = torch.zeros((B,), device=device, dtype=logits.dtype)

        # 5) Reference CE for gating
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

        # 7) TENT gating & loss（支持 EM-FT；样本级 gating）
        use_emft = bool(getattr(fa, "use_emft_loss", False))
        tent_base_vec = tent_seq_total_entropy if use_emft else tent_seq_entropy  # [B]

        if apply_tent_gate:
            mask_s = (ce_ref > threshold).to(logits.dtype)
            tent_vec = tent_base_vec * mask_s  # [B]
            tent_loss = (tent_vec.sum() / mask_s.sum().clamp_min(1.0)) if mask_s.sum() != 0 else tent_base_vec.mean()
        else:
            tent_loss = tent_base_vec.mean()

        # 8) Balancer (raw weights) then alternating (effective)
        base_w_ttl = float(getattr(fa, "loss_weight_ttl", 1.0))
        base_w_tent = float(getattr(fa, "loss_weight_tent", 1.0))
        ttl_selected_ratio = (
            float((ce_ref > threshold).float().mean().item()) if (apply_ttl_gate or apply_tent_gate) else 1.0
        )
        tent_valid_ratio = float(final_mask.mean().item()) if final_mask.numel() > 0 else 1.0
        context = {"ttl_selected_ratio": ttl_selected_ratio, "tent_valid_ratio": tent_valid_ratio}

        w_ttl_raw, w_tent_raw, extra_loss = self._balancer.compute_weights(
            self, base_w_ttl, base_w_tent, ttl_loss, tent_loss, context
        )

        alternating = bool(getattr(fa, "alternating_training", False))
        ttl_active = True
        tent_active = True
        if alternating:
            step_idx = (self.state.global_step or 0) + 1
            ttl_active = step_idx % 2 == 1
            tent_active = not ttl_active

        w_ttl_eff = w_ttl_raw if ttl_active else 0.0
        w_tent_eff = w_tent_raw if tent_active else 0.0

        # 9) reverse-KL(student‖teacher) on continuation only
        kl_loss = logits.new_zeros(())
        if bool(getattr(fa, "use_kl_regularization", False)) and generation_len != 0 and T >= 2:
            # KL 只作用在续写段有效 token 上（非 PAD 非 EOS）
            kl_mask_bool = final_mask > 0  # [B, T-1]
            flat_mask = kl_mask_bool.reshape(-1)
            valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_valid]
            if valid_idx.numel() > 0:
                # 学生端：复用已计算的 log_probs
                logp_s_flat = log_probs_all.reshape(-1, V)  # [B*(T-1), V]
                logp_s = logp_s_flat.index_select(0, valid_idx)  # [N, V]
                p_s = logp_s.exp()

                # 教师端：优先 disable_adapter()，否则回退到 self.teacher
                use_amp = bool(getattr(self.args, "fp16", False) or getattr(self.args, "bf16", False))
                unwrapped = getattr(self.accelerator, "unwrap_model", lambda x: x)(self.model)
                base_ctx_fn = getattr(unwrapped, "disable_adapter", None)
                base_ctx = base_ctx_fn() if base_ctx_fn is not None else _nullctx()

                with torch.inference_mode():
                    if use_amp and torch.cuda.is_available():
                        amp_dtype = torch.float16 if getattr(self.args, "fp16", False) else torch.bfloat16
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            if base_ctx_fn is not None:
                                with base_ctx:
                                    t_logits_all = self.model(
                                        input_ids=generated_tokens, attention_mask=attention_mask_gen
                                    ).logits  # [B, T, V]
                            elif self.teacher is not None:
                                t_logits_all = self.teacher(
                                    input_ids=generated_tokens, attention_mask=attention_mask_gen
                                ).logits
                            else:
                                t_logits_all = None
                    else:
                        if base_ctx_fn is not None:
                            with base_ctx:
                                t_logits_all = self.model(
                                    input_ids=generated_tokens, attention_mask=attention_mask_gen
                                ).logits
                        elif self.teacher is not None:
                            t_logits_all = self.teacher(
                                input_ids=generated_tokens, attention_mask=attention_mask_gen
                            ).logits
                        else:
                            t_logits_all = None

                if t_logits_all is not None:
                    t_logits_t1 = t_logits_all[:, :-1, :].reshape(-1, V)  # [B*(T-1), V]
                    t_logits = t_logits_t1.index_select(0, valid_idx)  # [N, V]
                    # 及时释放大张量
                    del t_logits_all
                    del t_logits_t1

                    logq_t = torch.log_softmax(t_logits, dim=-1)  # [N, V]
                    kl_tok = (p_s * (logp_s - logq_t)).sum(dim=-1)  # [N]
                    kl_loss = kl_tok.mean()

        # KL 不进入 balancer；若未启用则权重强制为 0
        kl_weight = float(getattr(fa, "kl_weight", 0.0)) if bool(getattr(fa, "use_kl_regularization", False)) else 0.0
        total_loss = (ttl_loss * w_ttl_eff) + (tent_loss * w_tent_eff) + (kl_weight * kl_loss) + extra_loss

        # 10) record history for plots (use raw per-branch losses and raw balancer weights)
        self._accumulate_for_step(ttl_loss.detach().item(), tent_loss.detach().item(), w_ttl_raw, w_tent_raw)

        # 11) token-level diagnostics
        try:
            with torch.no_grad():
                ex_ids = inputs.get("example_id", None)
                if isinstance(ex_ids, torch.Tensor):
                    ex_ids = ex_ids.detach().cpu().tolist()
                else:
                    ex_ids = [-1] * B

                prompt_ids_list, gen_ids_list = [], []
                prompt_nll_list, gen_entropy_list = [], []

                # y 对应 generated_tokens 的 [1:]，与 log_probs_all / entropy_tok 对齐
                y_tokens = generated_tokens[:, 1:]  # [B, T-1]

                for i in range(B):
                    # prompt tokens（按原 attn 截断）
                    if attn is not None:
                        plen = int(attn[i].sum().item())
                    else:
                        plen = int(prompt_max_len)
                    prompt_ids = input_ids[i, :plen].detach().cpu().tolist()
                    prompt_ids_list.append(prompt_ids)

                    # generation tokens（依据 final_mask）
                    if (T - 1) > 0:
                        mask_i = (final_mask[i] > 0).bool().detach().cpu()
                        gen_ids_i = y_tokens[i].detach().cpu()
                        gen_ids_i = gen_ids_i[mask_i] if mask_i.numel() > 0 else torch.tensor([], dtype=y_tokens.dtype)
                        gen_ids_list.append(gen_ids_i.tolist())
                    else:
                        gen_ids_list.append([])

                    # prompt token-level NLL
                    if ttl_token_nll.size(1) > 0:
                        mask_pi = ttl_mask[i].bool().detach().cpu()
                        nll_i = ttl_token_nll[i].detach().cpu()
                        nll_i = nll_i[mask_pi] if mask_pi.numel() > 0 else torch.tensor([])
                        prompt_nll_list.append([float(x) for x in nll_i.tolist()])
                    else:
                        prompt_nll_list.append([])

                    # generation token-level entropy
                    if entropy_tok.size(1) > 0:
                        mask_gi = (final_mask[i] > 0).bool().detach().cpu()
                        ent_i = entropy_tok[i].detach().cpu()
                        ent_i = ent_i[mask_gi] if mask_gi.numel() > 0 else torch.tensor([])
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
