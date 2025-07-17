from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

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
    r"""
    A self-contained Trainer for Test-Time Adaptation using EATA.
    Implements entropy minimization with sample selection and optional diversity filtering.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        compute_loss_func: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args
        self.eata_mt = None  # moving average for S_div(x)
        self.token_log: List[List[Dict[str, Union[str, float]]]] = []

    @override
    def compute_loss(self,
                     model,
                     inputs,
                     return_outputs=False,
                     num_items_in_batch=None):
        # Step 1: Generate output tokens
        self.model.eval()
        with torch.no_grad():
            if self.finetuning_args.generation_len > 0:
                generated_tokens = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.finetuning_args.generation_len,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    do_sample=False,
                    top_k=None,
                    top_p=None,
                    temperature=1.0,
                )
            else:
                generated_tokens = inputs["input_ids"]
        self.model.train()

        # Step 2: Forward pass on generated tokens
        outputs: CausalLMOutput = self.model(input_ids=generated_tokens)
        logits = outputs.logits

        # Step 3: Get logits and tokens for loss calculation
        if self.finetuning_args.generation_len > 0 and getattr(
                self.finetuning_args, "use_full_entropy_in_generation", False):
            target_logits = logits[:, :-1, :]  # [B, L_full-1, V]
            target_tokens = generated_tokens[:, 1:]  # [B, L_full-1]
        else:
            prompt_len = inputs["input_ids"].size(1)
            target_logits = logits[:, prompt_len - 1:-1, :]  # [B, L_gen, V]
            target_tokens = generated_tokens[:, prompt_len:]  # [B, L_gen]

        # Step 4: Token-wise entropy
        probs = torch.nn.functional.softmax(target_logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]

        # Step 5: Mask padding and compute sequence-level entropy
        entropy_mask = (
            target_tokens
            != self.processing_class.pad_token_id).float()  # [B, L]
        token_entropy = (entropy * entropy_mask).sum(dim=-1) / (
            entropy_mask.sum(dim=-1) + 1e-8)  # [B]

        if self.is_in_train and target_tokens.numel() > 0:
            self.token_log.append((
                target_tokens.detach().cpu(),
                entropy.detach().cpu(),
                entropy_mask.detach().cpu().bool(),
            ))

        # Step 6: EATA Sample Selection (S_ent)
        E0 = getattr(self.finetuning_args, "eata_entropy_threshold", 0.4)

        # Choose high/low entropy samples
        if getattr(self.finetuning_args, "eata_select_high_entropy", False):
            # 选择高熵样本进行优化 (熵越大，权重越高)
            sent_mask = (token_entropy >= E0).float()
            sent_weight = torch.exp(token_entropy - E0)
        else:
            # 选择低熵样本进行优化 (熵越小，权重越高)
            sent_mask = (token_entropy < E0).float()
            sent_weight = torch.exp(-(token_entropy - E0))

        sent_score = (sent_weight * sent_mask).detach()  # [B]

        # Step 7 (optional): S_div(x) - diversity filtering
        if getattr(self.finetuning_args, "eata_use_sdiv", False):
            avg_logits = target_logits.mean(dim=1)  # [B, V]
            avg_probs = torch.softmax(avg_logits, dim=-1)  # [B, V]

            if self.eata_mt is None:
                self.eata_mt = avg_probs.mean(dim=0).detach()
            else:
                alpha = getattr(self.finetuning_args, "eata_sdiv_momentum",
                                0.1)
                self.eata_mt = alpha * avg_probs.mean(
                    dim=0).detach() + (1 - alpha) * self.eata_mt

            cos_sim = torch.nn.functional.cosine_similarity(
                avg_probs, self.eata_mt.unsqueeze(0), dim=1)  # [B]
            sdiv_thresh = getattr(self.finetuning_args, "eata_sdiv_threshold",
                                  0.4)
            sdiv_mask = (cos_sim < sdiv_thresh).float()  # [B]

            sent_score = (sent_score * sdiv_mask).detach()

        # Step 8: Final weighted entropy loss
        entropy = entropy.view(entropy.size(0), -1)
        entropy_mask = entropy_mask.view(entropy_mask.size(0), -1)
        sent_score_expanded = sent_score.unsqueeze(1).expand_as(
            entropy)  # [B, L]

        loss = (entropy * entropy_mask * sent_score_expanded).sum() / (
            entropy_mask * sent_score_expanded).sum().clamp_min(1e-8)

        return (loss, outputs) if return_outputs else loss

    @override
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args,
                                                     self.finetuning_args)
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
