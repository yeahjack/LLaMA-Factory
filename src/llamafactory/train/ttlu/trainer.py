# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
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

    对齐官方实现：
      - 以输入 token 本身作为目标，sentence 级 CE 用作筛选分数（参考分布），
        训练损失使用 KL(logits || one-hot(target))，其数值等价于 CE。
      - mask: sentence_ce > ttl_threshold
      - 权重: ttl_sample_efficiency_scaler * exp(sentence_ce - ttl_threshold)
      - 归约: 被选样本上求平均；若一个都没选到，退化为简单平均（与官方等价）。
    参考分布两种模式：
      - ttl_ref_mode="precompute": 训练前一次性用 base model 计算并缓存每样本 CE（由 workflow 负责遍历与缓存）；
      - ttl_ref_mode="simultaneous": 训练时用当前模型即时计算 CE。
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args

        # 预计算的参考 sentence CE：dict[example_id] -> float (cpu)
        # 由 workflow 在训练开始前填充，不在 Trainer 内部构建 dataloader。
        self.ref_sentence_ce: Optional[dict[int, float]] = None

        # 训练阶段的样本级日志（可选）
        self._log_path: Optional[str] = None

    # ------------------------ 优化器 / 调度器 ------------------------ #
    @override
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    # ------------------------ 核心损失实现 -------------------------- #
    @override
    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # 不使用上游监督标签（TTL 自监督于输入）
        if "labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}

        input_ids: torch.Tensor = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn,
        )
        logits: torch.Tensor = outputs["logits"]  # [B, L, V]

        # —— 关键：屏蔽 pad 与句首 —— #
        labels_eff = input_ids.clone()
        if attn is not None:
            labels_eff = labels_eff.masked_fill(attn == 0, IGNORE_INDEX)
        labels_eff[:, 0] = IGNORE_INDEX

        # 训练损失（与官方等价）：KL(logits || one-hot(labels_eff))
        sentence_kl = self._cal_kl(logits, labels_eff)  # [B]

        # 参考 CE 来源
        ref_mode = getattr(self.finetuning_args, "ttl_ref_mode", "precompute").lower()
        if ref_mode not in {"precompute", "simultaneous"}:
            raise ValueError(f"Unsupported ttl_ref_mode: {ref_mode}")

        if ref_mode == "simultaneous":
            with torch.no_grad():
                sentence_ce = self._cal_ce(logits, labels_eff)  # [B]
        else:
            if "example_id" not in inputs:
                raise RuntimeError(
                    "ttl_ref_mode=precompute 需要 batch 中包含 example_id；"
                    "请在 workflow 中为数据集添加该字段，并用包装 collator 传递。"
                )
            if self.ref_sentence_ce is None:
                raise RuntimeError("参考 CE 尚未预计算，请先在 workflow 中完成预计算并设置 trainer.ref_sentence_ce。")
            ex_ids = inputs["example_id"]
            if isinstance(ex_ids, torch.Tensor):
                ex_ids = ex_ids.tolist()
            sentence_ce = torch.tensor(
                [self.ref_sentence_ce[int(e)] for e in ex_ids],
                dtype=logits.dtype,
                device=logits.device,
            )

        # gating 与权重
        threshold = float(getattr(self.finetuning_args, "ttl_threshold", 3.0))
        scaler = float(getattr(self.finetuning_args, "ttl_sample_efficiency_scaler", 0.1))

        mask = (sentence_ce > threshold).to(logits.dtype)  # [B]
        coeff = scaler * torch.exp(sentence_ce.detach() - threshold)  # [B]

        weighted = sentence_kl * coeff * mask  # [B]
        if mask.sum() == 0:
            total_loss = weighted.mean()
        else:
            total_loss = weighted.sum() / mask.sum()

        # 可选样本级日志
        if self._log_path is not None and self.is_world_process_zero():
            with open(self._log_path, "a", encoding="utf-8") as f:
                for ce_val, kl_val, m_val, c_val in zip(
                    sentence_ce.detach().tolist(),
                    sentence_kl.detach().tolist(),
                    mask.detach().tolist(),
                    coeff.detach().tolist(),
                ):
                    if m_val > 0:
                        print(
                            f"Selected. threshold={threshold}, CE={ce_val:.6f}, KL={kl_val:.6f}, "
                            f"coeff={c_val:.6f}, loss={total_loss.item():.6f}",
                            file=f,
                        )
                    else:
                        print(
                            f"Discarded. threshold={threshold}, CE={ce_val:.6f}, KL={kl_val:.6f}",
                            file=f,
                        )

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------ 辅助：CE / KL ------------------------- #
    @torch.no_grad()
    def _cal_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """对输入 tokens 计算 sentence 级交叉熵（忽略 IGNORE_INDEX），等价于负对数似然。"""
        criterion = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        shift_logits = logits[..., :-1, :]  # [B, L-1, V]
        shift_labels = labels[..., 1:]  # [B, L-1]
        loss = criterion(
            shift_logits.contiguous().view(-1, shift_logits.size(-1)),
            shift_labels.contiguous().view(-1),
        ).view(shift_labels.size())  # [B, L-1]
        mask = (shift_labels != IGNORE_INDEX).to(loss.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        sent_ce = (loss * mask).sum(dim=1) / denom
        return sent_ce  # [B]

    def _cal_kl(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """KLDivLoss(log_probs || one_hot(labels)) 的句子级平均（对有效 token）。数值等价于 CE。"""
        loss_fct = nn.KLDivLoss(reduction="batchmean")

        shift_logits = logits[..., :-1, :]  # [B, L-1, V]
        shift_labels = labels[..., 1:]  # [B, L-1]
        B = shift_logits.size(0)
        device = shift_logits.device

        sentence_kl = torch.zeros(B, device=device, dtype=shift_logits.dtype)
        for i in range(B):
            mask = shift_labels[i] != IGNORE_INDEX
            if mask.any():
                log_probs = shift_logits[i][mask].log_softmax(dim=-1)  # [T, V]
                target_ids = shift_labels[i][mask]  # [T]
                one_hot = torch.zeros_like(log_probs).scatter_(1, target_ids.unsqueeze(1), 1.0)
                sentence_kl[i] = loss_fct(log_probs, one_hot)
            else:
                sentence_kl[i] = 0.0
        return sentence_kl  # [B]
