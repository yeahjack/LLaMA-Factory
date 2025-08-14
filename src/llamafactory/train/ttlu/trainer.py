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

import math
from typing import TYPE_CHECKING, Callable, Optional  # Callable 保留兼容（未使用）

import torch
from torch.nn import CrossEntropyLoss
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class TTLTrainer(Seq2SeqTrainer):
    """Seq2Seq Trainer for Test-Time Learning (TTL).

    与最初版本不同：TTL 损失（基于 NLL/PPL + 可选样本选择）在本类 `compute_loss()` 内部原生实现。
    `compute_loss_func` 参数仅保留以兼容旧调用；实际被忽略。
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        compute_loss_func: Optional[Callable] = None,  # deprecated, ignored
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args
        # token-level logging: list of (shift_labels[B,L], per_token_nll[B,L], mask_bool[B,L])
        self.token_log = []

    @override
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs):
        if self.finetuning_args.disable_shuffling:
            import torch as _torch

            return _torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # TTL 损失实现
    #
    # 配置字符串: "<metric>" 或 "<metric>_<selection>"
    #
    #   metric    ∈ {"nll","ppl"} : 实际优化的序列级度量（梯度来源）
    #   selection ∈ {"nll","ppl"} : 可选；若提供则启用样本选择 gating；
    #                               阈值比较在 selection 空间中完成。
    #
    #   示例：
    #       ttl_loss="nll"        -> 优化 NLL；无 selection。
    #       ttl_loss="ppl"        -> 优化 PPL；无 selection（不推荐大模型，梯度噪声大）。
    #       ttl_loss="nll_ppl"    -> 优化 NLL；用 PPL 做 gating（高困惑度样本）。
    #       ttl_loss="ppl_nll"    -> 优化 PPL；用 NLL gating。
    #       ttl_loss="nll_nll"    -> 优化 NLL；用 NLL gating。
    #
    # 返回值：标量平均损失（符合 HF Trainer 期望），无需随 batch-size 调整 LR。
    # ------------------------------------------------------------------ #
    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # --- Forward ---------------------------------------------------------- #
        # TTL 自定义损失，不需要模型内部计算 supervised loss，因此去掉 labels 等监督键。
        if "labels" in inputs:
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        else:
            model_inputs = inputs
        outputs = model(**model_inputs)
        logits = outputs["logits"]  # [B, L, V]

        # 获取原始输入序列 (shift labels)
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
        else:  # 兜底（workflow monkey-patch）
            input_ids = outputs.get("input_ids", None)
            if input_ids is None:
                raise RuntimeError("TTLTrainer.compute_loss: 无 input_ids；请确保 collator 传入 inputs['input_ids'].")

        # -------- per-token NLL ------------------------------------------------ #
        shift_logits = logits[..., :-1, :].contiguous()  # [B, L-1, V]
        shift_labels = input_ids[..., 1:].contiguous()  # [B, L-1]

        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())  # [B, L-1]

        # 保存未 mask CE 作为 NLL（供 token_log 导出）
        per_token_nll_unmasked = per_token_loss

        mask = shift_labels != IGNORE_INDEX  # [B, L-1]
        per_token_loss = per_token_loss * mask

        token_counts = mask.sum(dim=1).clamp(min=1)  # [B]
        per_sample_nll = per_token_loss.sum(dim=1) / token_counts
        per_sample_ppl = torch.exp(per_sample_nll.clamp(max=20))  # 防 overflow

        # -------- 解析 ttl_loss 字符串 ----------------------------------------- #
        ttl_loss_cfg = getattr(self.finetuning_args, "ttl_loss", "ppl_ppl")
        parts = ttl_loss_cfg.split("_")
        if not (1 <= len(parts) <= 2):
            raise ValueError(f"ttl_loss 必须形如 '<metric>' 或 '<metric>_<selection>'，收到: {ttl_loss_cfg}")

        metric_form = parts[0].lower()
        selection_form = parts[1].lower() if len(parts) == 2 else None

        if metric_form not in {"nll", "ppl"}:
            raise ValueError(f"Unsupported metric_form: {metric_form}")
        if selection_form is not None and selection_form not in {"nll", "ppl"}:
            raise ValueError(f"Unsupported selection_form: {selection_form}")

        # -------- selection gating (optional) ---------------------------------- #
        selection_score = None
        if selection_form is not None:
            # scaler / threshold
            scaler = getattr(self.finetuning_args, "ttl_sample_efficiency_scaler", 1.0)
            base_threshold = getattr(self.finetuning_args, "ttl_threshold", 3.0)

            if selection_form == "ppl":
                ppl_threshold = math.exp(base_threshold)  # interpret base_threshold as log-ppl
                indicator = (per_sample_ppl > ppl_threshold).float()
                raw_score = (per_sample_ppl / ppl_threshold).clamp(max=1e6)
            else:  # "nll"
                nll_threshold = base_threshold
                indicator = (per_sample_nll > nll_threshold).float()
                raw_score = (per_sample_nll / nll_threshold).clamp(max=1e6)

            selection_score = (scaler * raw_score * indicator).detach()  # [B]

            num_selected = int(indicator.sum().item())
            total = per_sample_ppl.size(0)
            logger.info_rank0(f"[TTL] Selected {num_selected} / {total} high-perplexity samples.")

        # -------- metric for optimization -------------------------------------- #
        per_sample_metric = per_sample_ppl if metric_form == "ppl" else per_sample_nll

        if selection_score is None:
            ttl_loss = per_sample_metric.mean()
        else:
            ttl_loss = (selection_score * per_sample_metric).mean()

        # -------- token-level logging ------------------------------------------ #
        if self.is_in_train and shift_labels.numel() > 0:
            self.token_log.append(
                (
                    shift_labels.detach().cpu(),  # token ids
                    per_token_nll_unmasked.detach().cpu(),  # raw CE
                    mask.detach().cpu().bool(),  # valid positions
                )
            )

        return (ttl_loss, outputs) if return_outputs else ttl_loss
