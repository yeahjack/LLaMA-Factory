# src/llmtuner/train/tent/trainer.py

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


    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        r"""
        Computes the TENT loss. This method completely replaces the standard
        loss computation when `do_tent_adaptation` is enabled.
        """
        # if not self.finetuning_args.do_tent_adaptation:
        #     return super().compute_loss(model, inputs, return_outputs)

        # --- TENT Adaptation Logic (Self-Contained) ---

        # Step 1: Generate output tokens in evaluation mode (no dropout).
        # This step does not compute gradients.
        self.model.eval()
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.finetuning_args.tent_generation_len,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
                do_sample=False,
                top_k=None,  # Ensure deterministic generation
                top_p=None,
                temperature=1.0,  # No temperature scaling for TENT
            )
        self.model.train()  # IMPORTANT: Switch back to training mode for gradient computation

        # Step 2: Get logits for the generated tokens.
        # This forward pass WILL compute gradients.
        outputs: CausalLMOutput = self.model(input_ids=generated_tokens)
        logits = outputs.logits

        # Step 3: Isolate logits and tokens corresponding to the generated part.
        prompt_len = inputs["input_ids"].size(1)
        gen_logits = logits[:, prompt_len - 1:-1, :]
        gen_tokens = generated_tokens[:, prompt_len:]

        # Step 4: Calculate token-wise entropy from logits.
        probs = torch.nn.functional.softmax(gen_logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Step 5: Mask out padding tokens and compute the final mean entropy loss.
        entropy_mask = (gen_tokens != self.processing_class.pad_token_id).float()
        loss = (entropy * entropy_mask).sum() / (entropy_mask.sum() + 1e-8)

        # The base class expects a tuple if return_outputs is True.
        # For TENT, the "outputs" are the ones from the second forward pass.
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