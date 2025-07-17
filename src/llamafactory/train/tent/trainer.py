# src/llmtuner/train/tent/trainer.py

import torch
from transformers import Seq2SeqTrainer
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class TentTrainer(Seq2SeqTrainer):
    r"""
    A self-contained Trainer for Test-Time Entropy Minimization (TENT).

    This trainer overrides the `compute_loss` method to implement the TENT
    algorithm directly, ensuring perfect compatibility with the parent
    Trainer's calling conventions. The logic is encapsulated here rather
    than injected from the workflow.
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
        self.token_log: List[List[Dict[str, Union[str,
                                                  float]]]] = []  # noqa: F821

    @override
    def compute_loss(self,
                     model,
                     inputs,
                     return_outputs=False,
                     num_items_in_batch=None):
        self.model.eval()
        with torch.no_grad():
            if self.finetuning_args.generation_len > 0:
                # Step 1a: Generate tokens (no gradient)
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
                prompt_len = inputs["input_ids"].size(1)
            else:
                # Step 1b: Skip generation; use input directly
                generated_tokens = inputs["input_ids"]
                prompt_len = 0  # full sequence used

        self.model.train()  # Restore training mode

        # Step 2: Forward pass with gradient tracking
        outputs: CausalLMOutput = self.model(input_ids=generated_tokens)
        logits = outputs.logits  # [B, T, V]

        # Step 3: Slice logits/tokens based on generation mode
        if self.finetuning_args.generation_len > 0:
            if getattr(self.finetuning_args, "use_full_entropy_in_generation", False):
                # 使用整个序列 entropy
                gen_logits = logits[:, :-1, :]
                gen_tokens = generated_tokens[:, 1:]
            else:
                # 使用生成部分 entropy（默认逻辑）
                gen_logits = logits[:, prompt_len - 1:-1, :]
                gen_tokens = generated_tokens[:, prompt_len:]
        else:
            gen_logits = logits[:, :-1, :]
            gen_tokens = generated_tokens[:, 1:]

        # Step 4: Compute entropy
        probs = torch.nn.functional.softmax(gen_logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]

        # Step 5: Mask padding
        entropy_mask = (gen_tokens
                        != self.processing_class.pad_token_id).float()
        loss = (entropy * entropy_mask).sum() / (entropy_mask.sum() + 1e-8)

        if self.is_in_train and gen_tokens.numel() > 0:
            # 将一个批次的 tokens, entropies, 和 mask 作为一个元组存储
            # .detach().cpu() 是关键：分离计算图并移至CPU，以防GPU内存泄漏
            self.token_log.append((
                gen_tokens.detach().cpu(),
                entropy.detach().cpu(),
                entropy_mask.detach().cpu().bool(),  # 使用 bool mask 更高效
            ))

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
