# src/llmtuner/train/eata/trainer.py

import torch
from transformers import Seq2SeqTrainer
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override
from typing import TYPE_CHECKING, Callable, Optional

from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class EataTrainer(Seq2SeqTrainer):
    r"""
    A self-contained Trainer for Test-Time Entropy Minimization (EATA with sample selection).
    This trainer overrides the `compute_loss` method to implement the adaptation logic.
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

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Step 1: Generate output tokens in evaluation mode (no dropout).
        self.model.eval()
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.finetuning_args.tent_generation_len, #TODO set max_new_tokens=30 when using Llama2-13B-chat following the paper
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=1.0,
            )
        self.model.train()

        # Step 2: Get logits for the generated tokens.
        outputs: CausalLMOutput = self.model(input_ids=generated_tokens)
        logits = outputs.logits

        # Step 3: Isolate logits and tokens corresponding to the generated part.
        prompt_len = inputs["input_ids"].size(1)
        gen_logits = logits[:, prompt_len - 1:-1, :]  # [B, L, V]
        gen_tokens = generated_tokens[:, prompt_len:]  # [B, L]

        # Step 4: Token-wise entropy from logits
        probs = torch.nn.functional.softmax(gen_logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]

        # Step 5: Mask out padding tokens
        entropy_mask = (gen_tokens != self.processing_class.pad_token_id).float()  # [B, L]
        token_entropy = (entropy * entropy_mask).sum(dim=-1) / (entropy_mask.sum(dim=-1) + 1e-8)  # [B]

        # ------------------- EATA: Sample Selection -------------------
        E0 = getattr(self.finetuning_args, "eata_entropy_threshold", 0.4)  # Default to 0.4 if not provided
        sent_weight = torch.exp(-(token_entropy - E0))  # [B]
        sent_mask = (token_entropy < E0).float()  # [B]
        sent_score = (sent_weight * sent_mask).detach()  # [B]

        # Step 6: Final loss computation (sample-adaptive weighted entropy loss)
        entropy = entropy.view(entropy.size(0), -1)  # [B, L]
        entropy_mask = entropy_mask.view(entropy_mask.size(0), -1)  # [B, L]
        sent_score_expanded = sent_score.unsqueeze(1).expand_as(entropy)  # [B, L]

        loss = (entropy * entropy_mask * sent_score_expanded).sum() / (entropy_mask * sent_score_expanded).sum().clamp_min(1e-8)

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