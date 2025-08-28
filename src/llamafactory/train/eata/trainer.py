from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
from transformers import Seq2SeqTrainer
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class EataTrainer(Seq2SeqTrainer):
    r"""A self-contained Trainer for Test-Time Adaptation using EATA.
    Implements entropy minimization with sample selection and optional diversity filtering.
    Supports EM-FT (path-total entropy) via finetuning_args.use_emft_loss.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        compute_loss_func: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        # 接收预计算续写与模式
        self.precomputed_predictions: Optional[list[list[int]]] = kwargs.pop("precomputed_predictions", None)
        self.gen_model_mode: str = kwargs.pop("gen_model_mode", getattr(finetuning_args, "gen_model", "simultaneous"))

        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args
        self.eata_mt = None  # moving average for S_div(x)
        self.token_log: list[list[dict[str, Union[str, float]]]] = []
        self._precompute_ptr: int = 0

        if hasattr(self, "processing_class") and self.processing_class is not None:
            try:
                self.processing_class.padding_side = "left"
                logger.info_rank0("Set tokenizer padding_side='left' for FlashAttention compatibility.")
            except Exception as e:
                logger.warning_rank0(f"Could not set padding_side: {e}")

    def _pad_and_stack(self, sequences: list[list[int]], pad_id: int, device, dtype) -> torch.Tensor:
        """
        Pad a list of token id lists to a tensor [B, Lmax] with pad_id.
        If sequences is empty, return an empty 2D tensor with correct dtype/device.
        """
        bsz = len(sequences)
        max_len = max((len(s) for s in sequences), default=0)
        if max_len == 0:
            return torch.full((bsz, 0), pad_id, dtype=dtype, device=device)
        out = torch.full((bsz, max_len), pad_id, dtype=dtype, device=device)
        for i, s in enumerate(sequences):
            if len(s) > 0:
                out[i, : len(s)] = torch.tensor(s, dtype=dtype, device=device)
        return out

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

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Step 1: Generate or build output tokens
        self.model.eval()
        pad_id, eos_id = self._get_pad_eos_ids()

        with torch.no_grad():
            if self.finetuning_args.generation_len != 0:
                # 预计算模式：直接拼接续写；否则用当前模型生成
                use_precompute = (
                    getattr(self, "gen_model_mode", getattr(self.finetuning_args, "gen_model", "simultaneous"))
                    == "precompute"
                    and self.precomputed_predictions is not None
                )

                if self.finetuning_args.generation_len == -1:
                    max_tokens = 2048  # 动态上限；模型可遇 EOS 早停
                else:
                    max_tokens = self.finetuning_args.generation_len

                if use_precompute:
                    bsz = inputs["input_ids"].size(0)
                    cont_list: list[list[int]] = []
                    for _ in range(bsz):
                        if self._precompute_ptr < len(self.precomputed_predictions):
                            seq = self.precomputed_predictions[self._precompute_ptr]
                            self._precompute_ptr += 1
                        else:
                            seq = []
                        if self.finetuning_args.generation_len is not None and self.finetuning_args.generation_len > 0:
                            seq = seq[: self.finetuning_args.generation_len]
                        cont_list.append(seq)

                    cont_tensor = self._pad_and_stack(
                        cont_list,
                        pad_id=pad_id,
                        device=inputs["input_ids"].device,
                        dtype=inputs["input_ids"].dtype,
                    )
                    generated_tokens = torch.cat([inputs["input_ids"], cont_tensor], dim=1)
                else:
                    generated_tokens = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        max_new_tokens=max_tokens,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                        do_sample=False,
                        top_k=None,
                        top_p=None,
                        temperature=1.0,
                    )
                prompt_len = inputs["input_ids"].size(1)
            else:
                generated_tokens = inputs["input_ids"]
                prompt_len = 0
        self.model.train()

        # Step 2: Forward pass on generated tokens
        outputs: CausalLMOutput = self.model(input_ids=generated_tokens)
        logits = outputs.logits

        # Step 3: Get logits and tokens for loss calculation（与 TENT 对齐的切片规则）
        if self.finetuning_args.generation_len != 0:
            if getattr(self.finetuning_args, "use_full_entropy_in_generation", False):
                target_logits = logits[:, :-1, :]  # [B, L_full-1, V]
                target_tokens = generated_tokens[:, 1:]  # [B, L_full-1]
            else:
                target_logits = logits[:, prompt_len - 1 : -1, :]  # [B, L_gen, V]
                target_tokens = generated_tokens[:, prompt_len:]  # [B, L_gen]
        else:
            target_logits = logits[:, :-1, :]
            target_tokens = generated_tokens[:, 1:]

        # Step 4: Token-wise entropy
        probs = torch.nn.functional.softmax(target_logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]

        # Step 5: Mask padding and compute sequence-level entropy统计
        entropy_mask = (target_tokens != pad_id).float()  # [B, L]
        # 序列平均熵（原行为）
        seq_mean_entropy = (entropy * entropy_mask).sum(dim=-1) / (entropy_mask.sum(dim=-1) + 1e-8)  # [B]
        # 路径总熵（EM-FT）
        seq_total_entropy = (entropy * entropy_mask).sum(dim=-1)  # [B]

        if self.is_in_train and target_tokens.numel() > 0:
            self.token_log.append(
                (
                    target_tokens.detach().cpu(),
                    entropy.detach().cpu(),
                    entropy_mask.detach().cpu().bool(),
                )
            )

        # Step 6: EATA Sample Selection (S_ent) 基于序列平均熵进行阈值/加权（保持原逻辑）
        # 为了不改变既有行为，这里的选择仍然用 seq_mean_entropy 作为样本难度度量
        E0 = getattr(self.finetuning_args, "eata_entropy_threshold", 0.4)

        if getattr(self.finetuning_args, "eata_select_high_entropy", False):
            sent_mask = (seq_mean_entropy >= E0).float()
            sent_weight = torch.exp(seq_mean_entropy - E0)
        else:
            sent_mask = (seq_mean_entropy < E0).float()
            sent_weight = torch.exp(-(seq_mean_entropy - E0))

        sent_score = (sent_weight * sent_mask).detach()  # [B]

        # Step 7 (optional): S_div(x) - diversity filtering
        if getattr(self.finetuning_args, "eata_use_sdiv", False):
            avg_logits = target_logits.mean(dim=1)  # [B, V]
            avg_probs = torch.softmax(avg_logits, dim=-1)  # [B, V]

            if self.eata_mt is None:
                self.eata_mt = avg_probs.mean(dim=0).detach()
            else:
                alpha = getattr(self.finetuning_args, "eata_sdiv_momentum", 0.1)
                self.eata_mt = alpha * avg_probs.mean(dim=0).detach() + (1 - alpha) * self.eata_mt

            cos_sim = torch.nn.functional.cosine_similarity(avg_probs, self.eata_mt.unsqueeze(0), dim=1)  # [B]
            sdiv_thresh = getattr(self.finetuning_args, "eata_sdiv_threshold", 0.4)
            sdiv_mask = (cos_sim < sdiv_thresh).float()  # [B]

            sent_score = (sent_score * sdiv_mask).detach()

        # Step 8: Final entropy loss
        # 新增 EM-FT 选择：use_emft_loss=True -> 样本路径总熵按样本权重平均
        use_emft = bool(getattr(self.finetuning_args, "use_emft_loss", False))

        if use_emft:
            # EM-FT：对每个样本的“路径总熵”先求和，再对样本加权平均
            # 与 TentTrainer 的 EM-FT 一致：不按 token 数归一（避免稀释长续写的路径信息）
            num = (seq_total_entropy * sent_score).sum()
            den = sent_score.sum().clamp_min(1e-8)
            loss = num / den
        else:
            # 保持原实现：按 token 级别加样本权重，最后除以（加权的有效 token 总数）
            entropy_flat = entropy.view(entropy.size(0), -1)
            mask_flat = entropy_mask.view(entropy_mask.size(0), -1)
            sent_score_expanded = sent_score.unsqueeze(1).expand_as(entropy_flat)  # [B, L]

            loss = (entropy_flat * mask_flat * sent_score_expanded).sum() / (
                (mask_flat * sent_score_expanded).sum().clamp_min(1e-8)
            )

        return (loss, outputs) if return_outputs else loss

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
