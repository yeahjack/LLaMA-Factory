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

from typing import TYPE_CHECKING, Optional, List

import torch
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


class TTLTENTTrainer(Seq2SeqTrainer):
    """Seq2Seq Trainer for the combined TTL + TENT adaptation.

    TTL 部分（与 TTLU 对齐的结构，训练损失改为句子级 CE）：
      - 以输入 token 本身作为目标，训练损失为句子级 CE（忽略 padding 与句首）。
      - 参考分布（用于筛选与加权）支持：
          ttl_ref_mode="precompute": 训练前由 workflow 用 base model 预计算每样本 CE；
          ttl_ref_mode="simultaneous": 训练时用当前模型即时计算 CE。
      - gating 与权重：
          mask = 1{ CE_ref > ttl_threshold }
          coeff = ttl_sample_efficiency_scaler * exp(CE_ref - ttl_threshold)
          在被选样本上平均；若一个都没选到，退化为简单平均。

    TENT 部分（熵最小化）：
      - gen_model="simultaneous": 现场 generate；
      - gen_model="precompute": 使用 workflow 载入的 jsonl 预测，按 batch 顺序消费。
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        *args,
        **kwargs,
    ):
        # 供 TENT 使用的预计算续写序列与模式
        self.precomputed_predictions: Optional[List[List[int]]] = kwargs.pop("precomputed_predictions", None)
        self.gen_model_mode: str = kwargs.pop("gen_model_mode", getattr(finetuning_args, "gen_model", "simultaneous"))

        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args

        # TTL: 预计算的参考 sentence CE（仅在 ttl_ref_mode=precompute 时使用）
        self.ref_sentence_ce: Optional[dict[int, float]] = None

        # TENT: 预计算序列的消费指针
        self._precompute_ptr: int = 0

        # 训练阶段的可选 token 级日志
        self.token_log = []

        # decoder-only 常见：左填充以适配 FlashAttention
        if hasattr(self, "processing_class") and self.processing_class is not None:
            try:
                self.processing_class.padding_side = "left"
                logger.info_rank0("Set tokenizer padding_side='left' for FlashAttention compatibility.")
            except Exception as e:
                logger.warning_rank0(f"Could not set padding_side: {e}")

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

    @override
    def _get_train_sampler(self, *args, **kwargs):
        if getattr(self.finetuning_args, "disable_shuffling", False):
            import torch as _torch

            return _torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    # ------------------------ 核心：TTL + TENT ----------------------- #
    @override
    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # 不使用上游监督标签
        if "labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}

        # === 公共输入 ===
        input_ids: torch.Tensor = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)
        device = input_ids.device
        pad_id, eos_id = self._get_pad_eos_ids()

        ttl_weight = float(getattr(self.finetuning_args, "loss_weight_ttl", 1.0))
        tent_weight = float(getattr(self.finetuning_args, "loss_weight_tent", 1.0))
        generation_len = int(getattr(self.finetuning_args, "generation_len", 0))

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        outputs = None

        # =============== TTL: 句子级 CE + gating（与 TTLU 对齐） =============== #
        # 前向（输入序列）
        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits: torch.Tensor = outputs["logits"]  # [B, L, V]

        # 屏蔽 pad 与句首
        labels_eff = input_ids.clone()
        if attn is not None:
            labels_eff = labels_eff.masked_fill(attn == 0, IGNORE_INDEX)
        labels_eff[:, 0] = IGNORE_INDEX

        # 训练用 CE（句子级）
        sentence_ce_train = self._cal_ce(logits, labels_eff)  # [B]

        # 参考 CE 来源
        ref_mode = str(getattr(self.finetuning_args, "ttl_ref_mode", "precompute")).lower()
        if ref_mode not in {"precompute", "simultaneous"}:
            raise ValueError(f"Unsupported ttl_ref_mode: {ref_mode}")

        if ref_mode == "simultaneous":
            with torch.no_grad():
                sentence_ce_ref = self._cal_ce(logits, labels_eff)  # [B]
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
            sentence_ce_ref = torch.tensor(
                [self.ref_sentence_ce[int(e)] for e in ex_ids],
                dtype=logits.dtype,
                device=logits.device,
            )

        # gating 与权重
        threshold = float(getattr(self.finetuning_args, "ttl_threshold", 3.0))
        scaler = float(getattr(self.finetuning_args, "ttl_sample_efficiency_scaler", 0.1))

        mask = (sentence_ce_ref > threshold).to(logits.dtype)  # [B]
        coeff = scaler * torch.exp(sentence_ce_ref.detach() - threshold)  # [B]

        ttl_vec = sentence_ce_train * coeff * mask  # [B]
        if mask.sum() == 0:
            ttl_loss = ttl_vec.mean()
        else:
            ttl_loss = ttl_vec.sum() / mask.sum()

        total_loss = total_loss + ttl_weight * ttl_loss

        # ===================== TENT: 熵最小化 ====================== #
        if generation_len != 0 and tent_weight > 0.0:
            # 现场或预计算生成序列
            with torch.no_grad():
                if self.gen_model_mode == "precompute" and self.precomputed_predictions is not None:
                    cont = self._fetch_precomputed_continuations(
                        bsz=input_ids.size(0),
                        limit=max(0, generation_len) if generation_len > 0 else None,
                        pad_id=pad_id,
                        device=device,
                        dtype=input_ids.dtype,
                    )
                    generated_tokens = torch.cat([input_ids, cont], dim=1)
                else:
                    max_tokens = 2048 if generation_len == -1 else generation_len
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
                    )
                prompt_len = input_ids.size(1)

            # 前向（生成后的拼接序列）
            outputs_gen: CausalLMOutput = self.model(input_ids=generated_tokens)
            logits_gen = outputs_gen.logits  # [B, T, V]

            # 切片生成段
            if getattr(self.finetuning_args, "use_full_entropy_in_generation", False):
                gen_logits = logits_gen[:, :-1, :]
                gen_tokens = generated_tokens[:, 1:]
            else:
                gen_logits = logits_gen[:, prompt_len - 1 : -1, :]
                gen_tokens = generated_tokens[:, prompt_len:]

            # token-wise 熵
            log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]

            # mask 并归约
            entropy_mask = (gen_tokens != pad_id).float()
            if getattr(self.finetuning_args, "use_emft_loss", False):
                tent_loss = (entropy * entropy_mask).sum(dim=1).mean()
            else:
                tent_loss = (entropy * entropy_mask).sum() / (entropy_mask.sum().clamp_min(1e-8))

            total_loss = total_loss + tent_weight * tent_loss
        else:
            tent_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 训练时可选日志（频率由外部控制）
        if self.is_in_train and "input_ids" in inputs:
            # 可按需扩展 token 级日志；这里保留最小记录，避免显存负担
            pass

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------ 辅助：CE ------------------------- #
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

    # ------------------------ 辅助：TENT 预计算序列 ------------------------- #
    def _get_pad_eos_ids(self):
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
        return pad_id, eos_id

    def _pad_and_stack(self, sequences: List[List[int]], pad_id: int, device, dtype) -> torch.Tensor:
        bsz = len(sequences)
        max_len = max((len(s) for s in sequences), default=0)
        if max_len == 0:
            return torch.full((bsz, 0), pad_id, dtype=dtype, device=device)
        out = torch.full((bsz, max_len), pad_id, dtype=dtype, device=device)
        for i, s in enumerate(sequences):
            if len(s) > 0:
                out[i, : len(s)] = torch.tensor(s, dtype=dtype, device=device)
        return out

    def _fetch_precomputed_continuations(
        self, bsz: int, limit: Optional[int], pad_id: int, device, dtype
    ) -> torch.Tensor:
        """按 batch 顺序从 self.precomputed_predictions 中取出 bsz 条，裁剪到 limit，并 pad 成 [B, Lmax]。"""
        cont_list: List[List[int]] = []
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
