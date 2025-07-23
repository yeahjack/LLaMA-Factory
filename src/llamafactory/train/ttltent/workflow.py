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

import json
import os
from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import CombinedTTLTentTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)


def run_ttltent(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    Main workflow for combined TTL-TENT adaptation.

    This workflow combines:
    - TTL (Test-Time Learning) on input sequences
    - TENT (Test-Time Entropy Minimization) on generated sequences
    """
    # Load tokenizer and prepare template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load datasets
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="ttl",  # Use TTL stage for data loading
        **tokenizer_module,
    )

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

    # Disable generation for evaluation
    if training_args.predict_with_generate:
        logger.warning_once("`predict_with_generate` is not supported in combined TTL-TENT stage.")
        training_args.predict_with_generate = False

    # Monkey-patch forward to include input_ids in outputs
    orig_forward = model.forward

    def forward_with_ids(*args, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Remove labels to prevent supervised loss calculation in model
        if "labels" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
        outputs = orig_forward(
            *args,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        outputs["input_ids"] = input_ids
        return outputs

    model.forward = forward_with_ids

    # Initialize CombinedTTLTentTrainer
    trainer = CombinedTTLTentTrainer(
        finetuning_args=finetuning_args,
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
        # Log configuration
        logger.info_rank0("Starting Combined TTL-TENT Training with configuration:")
        logger.info_rank0(f"  TTL Loss: {getattr(finetuning_args, 'ttl_loss', 'nll')}")
        logger.info_rank0(f"  TTL Weight: {getattr(finetuning_args, 'loss_weight_ttl', 1.0)}")
        logger.info_rank0(f"  TENT Weight: {getattr(finetuning_args, 'loss_weight_tent', 1.0)}")
        logger.info_rank0(f"  Generation Length: {getattr(finetuning_args, 'generation_len', 0)}")

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # Post-processing: Save combined token analysis
        if trainer.is_world_process_zero():
            if hasattr(trainer, "token_log") and trainer.token_log:
                logger.info("Post-processing token logs for combined TTL-TENT analysis...")

                combined_log = []

                # Process each batch's data
                for batch_entry in trainer.token_log:
                    # Process input part (TTL)
                    if "input" in batch_entry:
                        input_data = batch_entry["input"]
                        input_tokens = input_data["tokens"]
                        input_nll = input_data["nll"]
                        input_mask = input_data["mask"]

                        # Process each sample in the batch
                        for i in range(input_tokens.size(0)):
                            sample_data = {"type": "combined", "input_analysis": [], "generated_analysis": []}

                            # Process input tokens
                            for j in range(input_tokens.size(1)):
                                if input_mask[i, j]:
                                    token_id = input_tokens[i, j].item()
                                    token_str = tokenizer.decode(token_id)
                                    nll_val = input_nll[i, j].item()

                                    sample_data["input_analysis"].append(
                                        {"token": token_str, "nll": round(float(nll_val), 6)}
                                    )

                            # Process generated tokens if available
                            if "generated" in batch_entry:
                                gen_data = batch_entry["generated"]
                                gen_tokens = gen_data["tokens"]
                                gen_entropy = gen_data["entropy"]
                                gen_mask = gen_data["mask"]

                                for j in range(gen_tokens.size(1)):
                                    if gen_mask[i, j]:
                                        token_id = gen_tokens[i, j].item()
                                        token_str = tokenizer.decode(token_id)
                                        entropy_val = gen_entropy[i, j].item()

                                        sample_data["generated_analysis"].append(
                                            {"token": token_str, "entropy": round(float(entropy_val), 6)}
                                        )

                            if sample_data["input_analysis"] or sample_data["generated_analysis"]:
                                combined_log.append(sample_data)

                # Save combined analysis
                output_file = os.path.join(training_args.output_dir, "combined_ttl_tent_analysis.json")
                logger.info(f"Saving combined analysis for {len(combined_log)} samples...")
                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(combined_log, f, indent=2, ensure_ascii=False)
                    logger.info(f"Combined TTL-TENT analysis successfully saved to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save combined analysis: {e}")

                # Save individual loss curves if available
                if hasattr(trainer, "_custom_losses"):
                    losses_file = os.path.join(training_args.output_dir, "loss_components.json")
                    loss_data = {
                        "ttl_loss": trainer._custom_losses.get("ttl_loss", []),
                        "tent_loss": trainer._custom_losses.get("tent_loss", []),
                        "steps": list(range(len(trainer._custom_losses.get("ttl_loss", [])))),
                    }
                    try:
                        with open(losses_file, "w") as f:
                            json.dump(loss_data, f, indent=2)
                        logger.info(f"Loss components saved to {losses_file}")
                    except Exception as e:
                        logger.error(f"Failed to save loss components: {e}")
            else:
                logger.warning("No token logs were collected, skipping analysis save.")

        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card and push
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
