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

from typing import TYPE_CHECKING, Optional

import torch
from torch.nn import CrossEntropyLoss

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import TTLTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)


def run_ttl(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # Load tokenizer and prepare template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load datasets for TTL
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="ttl",
        **tokenizer_module,
    )

    finetuning_args.ttl_perplexity_threshold = 20.086 # Default threshold for TTL, can be overridden by user, current value is e**3
    finetuning_args.ttl_sample_efficiency_scaler = 0.1  # Default scaling factor for TTL, can be overridden by user

    # Load model
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # Data collator
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=(IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Disable generation
    if training_args.predict_with_generate:
        logger.warning_once("`predict_with_generate` is not supported in TTL stage.")
        training_args.predict_with_generate = False

    # Monkey-patch forward to include input_ids in outputs
    orig_forward = model.forward
    def forward_with_ids(*args, input_ids=None, attention_mask=None, **kwargs):
        outputs = orig_forward(*args, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        outputs["input_ids"] = input_ids
        return outputs
    model.forward = forward_with_ids

    # Define TTL loss function matching signature (outputs, labels, num_items_in_batch)
    def compute_ttl_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"]
        input_ids = outputs["input_ids"]

        # Autoregressive shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Token-wise NLL
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        # Padding mask
        mask = shift_labels != IGNORE_INDEX
        per_token_loss = per_token_loss * mask

        # Avg NLL per sample
        per_sample_nll = per_token_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Perplexity = exp(NLL)
        per_sample_ppl = torch.exp(per_sample_nll.clamp(max=20))  # clamp to avoid overflow

        # Compute selection mask and score
        ppl_threshold = finetuning_args.ttl_perplexity_threshold  # P₀
        indicator = (per_sample_ppl > ppl_threshold).float()

        selection_score = (
            finetuning_args.ttl_sample_efficiency_scaler
            * (per_sample_ppl / ppl_threshold).clamp(max=1e6)
            * indicator
        ).detach()

        # Logging
        num_selected = int(indicator.sum().item())
        total = per_sample_ppl.size(0)
        logger.info_rank0(f"[TTL] Selected {num_selected} / {total} high-perplexity samples.")

        # Final loss = weighted perplexity
        ttl_loss = (selection_score * per_sample_ppl).sum() / (selection_score.sum() + 1e-8)

        return ttl_loss

    # Initialize TTLTrainer with custom loss
    trainer = TTLTrainer(
        finetuning_args=finetuning_args,
        compute_loss_func=compute_ttl_loss,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        train_dataset=dataset_module.get("train_dataset"),
        eval_dataset=dataset_module.get("eval_dataset"),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card and push
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
