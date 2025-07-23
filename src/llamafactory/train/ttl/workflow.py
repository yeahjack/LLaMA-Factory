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

    def forward_with_ids(*args, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # TTL 仅需 logits，不需要模型内部 supervised loss；屏蔽 labels 传递。
        if "labels" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
        # 也忽略显式形参 labels (若上游传入)
        outputs = orig_forward(
            *args,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        outputs["input_ids"] = input_ids
        return outputs

    model.forward = forward_with_ids

    # Initialize TTLTrainer (no external compute_loss_func; TTLTrainer handles loss internally)
    trainer = TTLTrainer(
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
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # ------------------------------------------------------------------ #
        # 后处理与保存 token->PPL 统计 (仿 Tent/EATA)
        # ------------------------------------------------------------------ #
        if trainer.is_world_process_zero():
            if hasattr(trainer, "token_log") and trainer.token_log:
                logger.info("Post-processing token logs to generate PPL details...")

                final_log = []
                # 遍历每个批次的原始数据 (tokens, nll, mask)
                for batch_tokens, batch_nll, batch_mask in trainer.token_log:
                    # 遍历批次中的每个样本
                    for i in range(batch_tokens.size(0)):
                        sample_details = []
                        # 遍历序列中的每个 token
                        for j in range(batch_tokens.size(1)):
                            if batch_mask[i, j]:  # 有效 token (非填充/IGNORE_INDEX)
                                token_id = batch_tokens[i, j].item()
                                token_str = tokenizer.decode(token_id)
                                nll_val = batch_nll[i, j].item()
                                sample_details.append(
                                    {
                                        "token": token_str,
                                        "nll": round(float(nll_val), 6),
                                    }
                                )

                        if sample_details:
                            final_log.append(sample_details)

                output_file = os.path.join(training_args.output_dir, "token_ppl_details.json")
                logger.info(f"Saving processed token-PPL details for {len(final_log)} samples...")
                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(final_log, f, indent=2, ensure_ascii=False)
                    logger.info(f"Token-PPL details successfully saved to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save token-PPL details: {e}")
            else:
                logger.warning("No raw token logs were collected, skipping PPL save.")

        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card and push
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
