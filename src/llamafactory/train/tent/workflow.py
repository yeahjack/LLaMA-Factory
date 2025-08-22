# src/llmtuner/train/tent/workflow.py

from typing import TYPE_CHECKING, Optional, List

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import TentTrainer  # <-- Import the self-contained trainer
import os
import json

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)


def run_tent(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    r"""
    Main workflow for TENT adaptation.
    This workflow uses the specialized TentTrainer which encapsulates the TENT logic.
    """
    # Load tokenizer, model, and dataset
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="tent", **tokenizer_module)
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

    # gen_model: "simultaneous" or "precompute"
    gen_model_mode = getattr(finetuning_args, "gen_model", "simultaneous")
    precomputed_predictions = None

    if gen_model_mode == "precompute":
        # ensure deterministic order before dataloader is built in Trainer
        try:
            if not getattr(finetuning_args, "disable_shuffling", False):
                finetuning_args.disable_shuffling = True
                logger.info_rank0("Set finetuning_args.disable_shuffling=True for precompute alignment.")
        except Exception as e:
            logger.warning_rank0(f"Could not set disable_shuffling: {e}")

        # dataset name: use data_args.dataset[0] as you showed
        try:
            if hasattr(data_args, "dataset") and isinstance(data_args.dataset, list) and len(data_args.dataset) > 0:
                dataset_name = data_args.dataset[0]
            else:
                dataset_name = "train_dataset"
        except Exception as e:
            logger.warning_rank0(f"Failed to read data_args.dataset, fallback to default. detail={e}")
            dataset_name = "train_dataset"

        precompute_dir = getattr(finetuning_args, "precompute_results", "results_qwen25_7b/base_sys_prompt/")
        precompute_path = os.path.join(precompute_dir, f"{dataset_name}.jsonl")
        logger.info_rank0(f"Precompute mode on. Loading predictions from: {precompute_path}")

        loaded_token_ids = []
        try:
            with open(precompute_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        text = rec.get("predict", "")
                        if not isinstance(text, str):
                            text = str(text)
                        token_ids = tokenizer.encode(text, add_special_tokens=False)
                        loaded_token_ids.append(token_ids)
                    except Exception as ie:
                        logger.warning_rank0(f"Skip a bad jsonl line: {ie}")

            logger.info_rank0(f"Loaded {len(loaded_token_ids)} precomputed continuations.")
            _train_ds = dataset_module.get("train_dataset")
            try:
                _train_len = len(_train_ds) if _train_ds is not None else None
            except Exception:
                _train_len = None
            if _train_len is not None and len(loaded_token_ids) < _train_len:
                logger.warning_rank0(
                    f"Precomputed lines ({len(loaded_token_ids)}) < train samples ({_train_len}); "
                    "empty continuations will be used for missing samples."
                )
            precomputed_predictions = loaded_token_ids
        except FileNotFoundError:
            logger.warning_rank0("Precompute file not found. Fallback to `simultaneous` generation.")
            gen_model_mode = "simultaneous"
            precomputed_predictions = None
        except Exception as e:
            logger.warning_rank0(f"Failed to load precompute file ({e}). Fallback to `simultaneous`.")
            gen_model_mode = "simultaneous"
            precomputed_predictions = None

    # Initialize our specialized TentTrainer
    trainer = TentTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        train_dataset=dataset_module.get("train_dataset"),
        eval_dataset=dataset_module.get("eval_dataset"),
        precomputed_predictions=precomputed_predictions,
        gen_model_mode=gen_model_mode,
    )

    # Start adaptation (training)
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # 后处理和保存
        if trainer.is_world_process_zero():
            # 检查是否有收集到的原始日志数据
            if hasattr(trainer, "token_log") and trainer.token_log:
                logger.info("Post-processing token logs to generate entropy details...")

                final_log = []
                # 遍历每个批次的原始数据 (tokens, entropies, mask)
                for batch_tokens, batch_entropies, batch_mask in trainer.token_log:
                    # 遍历批次中的每个样本
                    for i in range(batch_tokens.size(0)):
                        sample_details = []
                        # 遍历序列中的每个 token
                        for j in range(batch_tokens.size(1)):
                            if batch_mask[i, j]:  # 如果是有效 token (非填充)
                                token_id = batch_tokens[i, j].item()
                                token_str = tokenizer.decode(token_id)
                                entropy_val = batch_entropies[i, j].item()
                                sample_details.append({"token": token_str, "entropy": round(entropy_val, 4)})

                        if sample_details:
                            final_log.append(sample_details)

                output_file = os.path.join(training_args.output_dir, "token_entropy_details.json")
                logger.info(f"Saving processed token-entropy details for {len(final_log)} samples...")
                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(final_log, f, indent=2, ensure_ascii=False)
                    logger.info(f"Token-entropy details successfully saved to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save token-entropy details: {e}")
            else:
                logger.warning("No raw token logs were collected, skipping save.")

        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
