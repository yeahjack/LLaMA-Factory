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

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import TTLTENTTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)


def _add_example_id_column(ds):
    try:
        if "example_id" in ds.column_names:
            return ds
        return ds.add_column("example_id", list(range(len(ds))))
    except Exception:
        return ds


def _wrap_collator_with_ids(base_collator):
    def _fn(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = base_collator(features)
        if isinstance(batch, tuple):
            batch = batch[0]
        ids = []
        for f in features:
            if "example_id" not in f:
                raise RuntimeError("样本缺少 example_id，请确保数据集中包含该字段。")
            ids.append(int(f["example_id"]))
        batch["example_id"] = torch.tensor(ids, dtype=torch.long)
        return batch

    return _fn


def _predict_and_save_jsonl(trainer: TTLTENTTrainer, dataset, tokenizer, out_dir: str, gen_args) -> None:
    os.makedirs(out_dir, exist_ok=True)

    gen_kwargs = gen_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    pad_backup = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predict_with_generate_backup = trainer.args.predict_with_generate
    trainer.args.predict_with_generate = True

    preds = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)

    trainer.args.predict_with_generate = predict_with_generate_backup
    tokenizer.padding_side = pad_backup

    token_pad_id = tokenizer.pad_token_id
    label_ids = preds.label_ids
    pred_ids = preds.predictions

    for i in range(len(pred_ids)):
        arr = pred_ids[i]
        nz = (arr != token_pad_id).nonzero()
        if len(nz):
            first = nz[0]
            pred_ids[i] = arr[first:]

    decoded_inputs = tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(
        (label_ids if label_ids is not None else dataset["labels"]),
        skip_special_tokens=True,
    )
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    out_file = os.path.join(out_dir, "generated_predictions.jsonl")
    with open(out_file, "a", encoding="utf-8") as f:
        for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
            f.write(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False) + "\n")

    logger.info_rank0(f"Saved prediction results to {out_file}")


@torch.no_grad()
def _precompute_reference_ce(
    trainer: TTLTENTTrainer,
    dataset,
    batch_size: int,
    log_path: str | None = None,
) -> None:
    from contextlib import nullcontext as _nullctx

    assert hasattr(trainer, "data_collator"), "TTLTENTTrainer 需要 data_collator 才能预计算参考 CE。"
    model = trainer.model
    model.eval()

    old_pdev_bs = getattr(trainer.args, "per_device_eval_batch_size", None)
    try:
        trainer.args.per_device_eval_batch_size = max(1, int(batch_size))
        dataloader = trainer.get_eval_dataloader(dataset)
        if trainer.is_world_process_zero():
            logger.info(
                f"[TTLTENT] Precompute CE per-device batch size set to {trainer.args.per_device_eval_batch_size}."
            )
    finally:
        if old_pdev_bs is not None:
            trainer.args.per_device_eval_batch_size = old_pdev_bs

    base_model_ctx = getattr(trainer.accelerator.unwrap_model(model), "disable_adapter", None)
    base_ctx = base_model_ctx() if base_model_ctx is not None else _nullctx()

    trainer.ref_sentence_ce = {}
    total_seen = 0

    show_bar = trainer.is_world_process_zero()
    try:
        total_examples = len(dataset)
    except Exception:
        total_examples = None
    pbar = (
        tqdm(total=total_examples, dynamic_ncols=True, unit="ex", desc="[TTLTENT] Precompute CE", leave=True)
        if show_bar
        else None
    )

    with base_ctx:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs["logits"]

            if attn is not None:
                P = attn.size(1)
                ttl_logits = logits[:, : P - 1, :]
                ttl_labels = input_ids[:, 1:P]
                criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
                ttl_token_nll = criterion(
                    ttl_logits.reshape(-1, ttl_logits.size(-1)),
                    ttl_labels.reshape(-1),
                ).view(ttl_labels.size())
                ttl_mask = (attn[:, 1:P] != 0).to(ttl_token_nll.dtype)
                ttl_denom = ttl_mask.sum(dim=1).clamp_min(1.0)
                sent_ce = (ttl_token_nll * ttl_mask).sum(dim=1) / ttl_denom
            else:
                criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
                ttl_token_nll = criterion(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                ).view(input_ids.size(0), -1)
                ttl_mask = torch.ones_like(ttl_token_nll)
                ttl_denom = ttl_mask.sum(dim=1).clamp_min(1.0)
                sent_ce = (ttl_token_nll * ttl_mask).sum(dim=1) / ttl_denom

            ex_ids = batch["example_id"]
            if isinstance(ex_ids, torch.Tensor):
                ex_ids = ex_ids.tolist()
            for eid, val in zip(ex_ids, sent_ce.detach().cpu().tolist()):
                trainer.ref_sentence_ce[int(eid)] = float(val)

            bsz = len(ex_ids)
            total_seen += bsz
            if pbar is not None:
                if total_examples is not None:
                    pbar.update(bsz)
                    left = max(total_examples - total_seen, 0)
                    pbar.set_postfix({"seen": total_seen, "left": left, "bs": bsz})
                else:
                    pbar.update(1)
                    pbar.set_postfix({"seen": total_seen, "bs": bsz})

    if pbar is not None:
        pbar.close()

    if log_path is not None and trainer.is_world_process_zero():
        with open(log_path, "a", encoding="utf-8") as f:
            print(f"[TTLTENT] Precomputed reference CE for {total_seen} samples.", file=f)

    model.train()


def _load_precomputed_predictions_if_needed(
    tokenizer, data_args, finetuning_args, dataset_module
) -> tuple[str, list[list[int]] | None]:
    gen_model_mode = str(getattr(finetuning_args, "gen_model", "simultaneous")).lower()
    precomputed_predictions: list[list[int]] | None = None
    dataset_name = None

    if gen_model_mode == "precompute":
        try:
            if not getattr(finetuning_args, "disable_shuffling", False):
                finetuning_args.disable_shuffling = True
                logger.info_rank0("Set finetuning_args.disable_shuffling=True for precompute alignment.")
        except Exception as e:
            logger.warning_rank0(f"Could not set disable_shuffling: {e}")

        try:
            if hasattr(data_args, "dataset") and isinstance(data_args.dataset, list) and len(data_args.dataset) > 0:
                dataset_name = data_args.dataset[0]
            else:
                dataset_name = "train_dataset"
        except Exception as e:
            logger.warning_rank0(f"Failed to read data_args.dataset, fallback to default. detail={e}")
            dataset_name = "train_dataset"

        precompute_dir = getattr(finetuning_args, "precompute_results", None)
        if precompute_dir is None:
            logger.warning_rank0("finetuning_args.precompute_results is None; fallback to `simultaneous` generation.")
        else:
            precompute_path = os.path.join(precompute_dir, f"{dataset_name}.jsonl")
            logger.info_rank0(f"[TTLTENT] Precompute mode on. Loading predictions from: {precompute_path}")

            loaded_token_ids: list[list[int]] = []
            try:
                with open(precompute_path, encoding="utf-8") as f:
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
                            gen_len = int(getattr(finetuning_args, "generation_len", 0))
                            if gen_len > 0:
                                token_ids = token_ids[:gen_len]
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
            except Exception as e:
                logger.warning_rank0(f"Failed to load precompute file ({e}). Fallback to `simultaneous`.")

    return (dataset_name or "train_dataset", precomputed_predictions)


def run_ttltent(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    generating_args: GeneratingArguments,
    callbacks: list[TrainerCallback] | None = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="ttl",
        **tokenizer_module,
    )

    train_dataset = dataset_module.get("train_dataset")
    eval_dataset = dataset_module.get("eval_dataset") or train_dataset
    train_dataset = _add_example_id_column(train_dataset)
    eval_dataset = _add_example_id_column(eval_dataset)

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    tokenizer.padding_side = "right"
    training_args.remove_unused_columns = False
    base_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=None,
        label_pad_token_id=(IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    data_collator = _wrap_collator_with_ids(base_collator)

    if training_args.predict_with_generate:
        logger.warning_once("`predict_with_generate` is not supported in TTL-TENT stage.")
        training_args.predict_with_generate = False

    dataset_name, precomputed_predictions = _load_precomputed_predictions_if_needed(
        tokenizer=tokenizer,
        data_args=data_args,
        finetuning_args=finetuning_args,
        dataset_module=dataset_module,
    )
    gen_model_mode = getattr(finetuning_args, "gen_model", "simultaneous")

    trainer = TTLTENTTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        precomputed_predictions=precomputed_predictions,
        gen_model_mode=gen_model_mode,
    )

    if str(getattr(finetuning_args, "ttl_ref_mode", "precompute")).lower() == "precompute":
        log_path = os.path.join(training_args.output_dir, "ttltent_ttl_log.txt")
        _precompute_reference_ce(
            trainer=trainer,
            dataset=train_dataset,
            batch_size=int(getattr(finetuning_args, "ttl_ref_batch_size", 64)),
            log_path=log_path,
        )

    direct_infer: bool = bool(getattr(finetuning_args, "ttl_direct_inference", False))
    setting = getattr(finetuning_args, "ttl_setting", "offline_ttl").lower()
    if setting not in {"offline_ttl", "online_ttl"}:
        raise ValueError(f"Unsupported ttl_setting: {setting}")

    if setting == "offline_ttl":
        if training_args.do_train:
            logger.info_rank0("Starting TTLTENT (offline) training...")
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.save_model()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss"])

        if direct_infer:
            pred_out = os.path.join(
                training_args.output_dir,
                f"predict-temperature_{generating_args.temperature}-max_new_tokens_{generating_args.max_new_tokens}",
            )
            _predict_and_save_jsonl(trainer, eval_dataset, tokenizer, pred_out, generating_args)

    else:
        bs = int(getattr(finetuning_args, "ttl_streaming_batch_size", 100))
        n = len(train_dataset)
        num_batches = n // bs + (1 if n % bs != 0 else 0)
        base_out = training_args.output_dir

        for k in range(num_batches):
            start, end = k * bs, min((k + 1) * bs, n)
            logger.info_rank0(f"[TTLTENT] Processing streaming batch {k + 1}/{num_batches}: [{start}, {end})")

            sub_train = train_dataset.select(range(start, end))
            sub_eval = eval_dataset.select(range(start, end))

            trainer.train_dataset = sub_train
            trainer.eval_dataset = sub_eval

            sub_out = os.path.join(base_out, f"online_step_{k:04d}")
            trainer.args.output_dir = sub_out
            os.makedirs(sub_out, exist_ok=True)

            if direct_infer:
                pred_out = os.path.join(
                    sub_out,
                    f"predict-temperature_{generating_args.temperature}-max_new_tokens_{generating_args.max_new_tokens}",
                )
                _predict_and_save_jsonl(trainer, sub_eval, tokenizer, pred_out, generating_args)

            train_result = trainer.train(resume_from_checkpoint=None)
            trainer.save_model()
            trainer.log_metrics(f"train_stream_{k}", train_result.metrics)
            trainer.save_metrics(f"train_stream_{k}", train_result.metrics)
            trainer.save_state()

        trainer.args.output_dir = base_out

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 将 token_log 以 JSON 数组形式保存（与 TENT/EATA 保持一致的 .json 文件）
    if trainer.is_world_process_zero():
        try:
            out_file = os.path.join(training_args.output_dir, "token_entropy_details.json")
            rows = []
            for rec in trainer.token_log:
                prompt_text = tokenizer.decode(rec.get("prompt_token_ids", []), skip_special_tokens=True)
                gen_text = tokenizer.decode(rec.get("generated_token_ids", []), skip_special_tokens=True)
                rows.append(
                    {
                        "example_id": rec.get("example_id", -1),
                        "prompt_text": prompt_text,
                        "generated_text": gen_text,
                        "prompt_token_nll": rec.get("prompt_token_nll", []),
                        "generation_token_entropy": rec.get("generation_token_entropy", []),
                    }
                )
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            logger.info_rank0(f"Token-entropy details saved to {out_file}")
        except Exception as e:
            logger.warning_rank0(f"Failed to save token-entropy details: {e}")

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
