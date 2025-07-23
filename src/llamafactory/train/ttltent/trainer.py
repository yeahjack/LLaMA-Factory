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
from typing import TYPE_CHECKING

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


class CombinedTTLTentTrainer(Seq2SeqTrainer):
    """Combined TTL-TENT Trainer for Test-Time Learning.

    This trainer combines:
    - TTL (Test-Time Learning): Optimizes perplexity/NLL on input sequences
    - TENT (Test-Time Entropy Minimization): Minimizes entropy on generated sequences

    The combined approach leverages both methods' strengths:
    - TTL helps the model adapt to the input distribution
    - TENT reduces uncertainty in generation
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args

        # Combined token log: stores both perplexity and entropy information
        self.token_log = []

        # Initialize a counter for cumulative selected samples for logging.
        self.cumulative_selected_samples = 0

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

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # Get configuration parameters
        generation_len = getattr(self.finetuning_args, "generation_len", 0)
        ttl_weight = getattr(self.finetuning_args, "loss_weight_ttl", 1.0)
        tent_weight = getattr(self.finetuning_args, "loss_weight_tent", 1.0)

        # Extract input_ids
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
        else:
            raise RuntimeError("CombinedTrainer.compute_loss: No input_ids found in inputs.")

        # Initialize losses
        ttl_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        tent_loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        # Part 1: TTL Loss on Input Sequence
        # Forward pass for input sequence
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs["logits"]  # [B, L, V]

        # Compute TTL loss (perplexity/NLL based)
        shift_logits = logits[..., :-1, :].contiguous()  # [B, L-1, V]
        shift_labels = input_ids[..., 1:].contiguous()  # [B, L-1]

        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())  # [B, L-1]

        # Save unmasked NLL for logging
        per_token_nll_input = per_token_loss.clone()

        mask = shift_labels != IGNORE_INDEX  # [B, L-1]
        per_token_loss = per_token_loss * mask

        token_counts = mask.sum(dim=1).clamp(min=1)  # [B]
        per_sample_nll = per_token_loss.sum(dim=1) / token_counts
        per_sample_ppl = torch.exp(per_sample_nll.clamp(max=20))  # Prevent overflow

        # Parse TTL loss configuration
        ttl_loss_cfg = getattr(self.finetuning_args, "ttl_loss", "ppl_nll")
        parts = ttl_loss_cfg.split("_")
        metric_form = parts[0].lower()
        selection_form = parts[1].lower() if len(parts) == 2 else None

        # Initialize selection count for logging
        num_selected = 0
        total_samples = per_sample_ppl.size(0)

        # Apply selection gating if configured
        selection_score = None
        if selection_form is not None:
            scaler = getattr(self.finetuning_args, "ttl_sample_efficiency_scaler", 1.0)
            base_threshold = getattr(self.finetuning_args, "ttl_threshold", 3.0)

            if selection_form == "ppl":
                ppl_threshold = math.exp(base_threshold)
                indicator = (per_sample_ppl > ppl_threshold).float()
                raw_score = (per_sample_ppl / ppl_threshold).clamp(max=1e6)
            else:  # "nll"
                nll_threshold = base_threshold
                indicator = (per_sample_nll > nll_threshold).float()
                raw_score = (per_sample_nll / nll_threshold).clamp(max=1e6)

            selection_score = (scaler * raw_score * indicator).detach()  # [B]

            num_selected = int(indicator.sum().item())
            logger.info_rank0(f"[TTL] Selected {num_selected} / {total_samples} high-perplexity samples.")

        # Compute TTL loss
        per_sample_metric = per_sample_ppl if metric_form == "ppl" else per_sample_nll

        if selection_score is None:
            ttl_loss = per_sample_metric.mean()
        else:
            ttl_loss = (selection_score * per_sample_metric).mean()

        # Part 2: TENT Loss on Generated Sequence (if generation_len > 0)
        generated_tokens = None
        entropy = None
        entropy_mask = None

        if generation_len > 0:
            # Generate tokens without gradient
            self.model.eval()
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=generation_len,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    do_sample=False,
                    top_k=None,
                    top_p=None,
                    temperature=1.0,
                )
                prompt_len = input_ids.size(1)

            self.model.train()  # Restore training mode

            # Forward pass on generated sequence with gradient
            gen_outputs: CausalLMOutput = self.model(input_ids=generated_tokens)
            gen_logits = gen_outputs.logits  # [B, T, V]

            # Slice logits/tokens for generation part
            if getattr(self.finetuning_args, "use_full_entropy_in_generation", False):
                # Use entropy from full sequence
                entropy_logits = gen_logits[:, :-1, :]
                entropy_tokens = generated_tokens[:, 1:]
            else:
                # Use entropy only from generated part (default)
                entropy_logits = gen_logits[:, prompt_len - 1 : -1, :]
                entropy_tokens = generated_tokens[:, prompt_len:]

            # Compute entropy
            probs = torch.nn.functional.softmax(entropy_logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(entropy_logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]

            # Mask padding
            entropy_mask = (entropy_tokens != self.processing_class.pad_token_id).float()

            # Compute TENT loss
            if getattr(self.finetuning_args, "use_emft_loss", False):
                # EM-FT Loss: Mean of Path-Total Entropy
                loss_per_sequence = (entropy * entropy_mask).sum(dim=1)
                tent_loss = loss_per_sequence.mean()
            else:
                # TENT Loss (Default): Batch-Average Token Entropy
                tent_loss = (entropy * entropy_mask).sum() / (entropy_mask.sum() + 1e-8)

        # Combine losses
        total_loss = ttl_weight * ttl_loss + tent_weight * tent_loss

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            # Update cumulative counter
            if self.is_in_train:
                self.cumulative_selected_samples += num_selected
            logger.info_rank0(
                f"[Combined] Step {self.state.global_step}: "
                f"TTL Loss = {ttl_loss.item():.4f}, "
                f"TENT Loss = {tent_loss.item():.4f}, "
                f"Total Loss = {total_loss.item():.4f}"
            )
            # Directly log custom metrics for wandb and other integrations
            self.log(
                {
                    "ttl_loss": ttl_loss.item(),
                    "tent_loss": tent_loss.item(),
                    "batch_avg_nll": per_sample_nll.mean().item(),
                    "batch_avg_ppl": per_sample_ppl.mean().item(),
                    "cumulative_selected_samples": float(self.cumulative_selected_samples),  # Added cumulative value
                }
            )

        # Token-level logging for analysis
        if self.is_in_train:
            log_entry = {
                "input": {
                    "tokens": shift_labels.detach().cpu(),
                    "nll": per_token_nll_input.detach().cpu(),
                    "mask": mask.detach().cpu().bool(),
                }
            }

            if generated_tokens is not None and entropy is not None:
                log_entry["generated"] = {
                    "tokens": entropy_tokens.detach().cpu(),
                    "entropy": entropy.detach().cpu(),
                    "mask": entropy_mask.detach().cpu().bool(),
                }

            self.token_log.append(log_entry)

        # Store individual losses for saving to loss_components.json (used by workflow.py)
        if not hasattr(self, "_custom_losses"):
            self._custom_losses = {"ttl_loss": [], "tent_loss": []}
        self._custom_losses["ttl_loss"].append(ttl_loss.item())
        self._custom_losses["tent_loss"].append(tent_loss.item())

        return (total_loss, outputs) if return_outputs else total_loss
