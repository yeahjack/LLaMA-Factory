# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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

import torch
from tqdm.auto import tqdm

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import TTLTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)


def _add_example_id_column(ds):
    """为 HuggingFace Dataset 添加 example_id 列，用于索引预计算 CE。"""
    try:
        if "example_id" in ds.column_names:
            return ds
        return ds.add_column("example_id", list(range(len(ds))))
    except Exception:
        # 兜底：IterableDataset 或不支持 add_column 的情况，退回原 ds（此时 precompute 会报错提示）
        return ds


def _wrap_collator_with_ids(base_collator):
    """包装 collator，把 example_id 拼进 batch；保持对原有键的完全透明。"""

    def _fn(features: list[dict]) -> dict:
        batch = base_collator(features)
        # 特殊情况：部分 collator 返回 tuple
        if isinstance(batch, tuple):
            batch = batch[0]
        # 从原始 features 抽取 example_id
        ids = []
        for f in features:
            if "example_id" not in f:
                raise RuntimeError("样本缺少 example_id，请确保数据集中包含该字段。")
            ids.append(int(f["example_id"]))
        batch["example_id"] = torch.tensor(ids, dtype=torch.long)
        return batch

    return _fn


def _predict_and_save_jsonl(trainer: TTLTrainer, dataset, tokenizer, out_dir: str, gen_args) -> None:
    """可选推理：对 dataset 生成并保存 JSONL（prompt/label/predict）."""
    os.makedirs(out_dir, exist_ok=True)

    # 生成参数
    gen_kwargs = gen_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # 生成阶段：decoder-only 需左填充
    pad_backup = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predict_with_generate_backup = trainer.args.predict_with_generate
    trainer.args.predict_with_generate = True

    preds = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
    # 还原设置
    trainer.args.predict_with_generate = predict_with_generate_backup
    tokenizer.padding_side = pad_backup

    # 将预测、输入、标签解码并落盘
    token_pad_id = tokenizer.pad_token_id
    label_ids = preds.label_ids
    pred_ids = preds.predictions

    # 顺序处理 pad：把左侧 pad 移到末尾，便于解码
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


def _precompute_reference_ce(
    trainer: TTLTrainer,
    dataset,
    batch_size: int,
    log_path: Optional[str] = None,
) -> None:
    """在 workflow 中执行参考 CE 的预计算：用 Trainer 的 eval dataloader 遍历 dataset，
    写入 trainer.ref_sentence_ce。"""
    from contextlib import nullcontext as _nullctx
    from tqdm.auto import tqdm

    assert hasattr(trainer, "data_collator"), "TTLTrainer 需要 data_collator 才能预计算参考 CE。"
    model = trainer.model
    model.eval()

    # 关键：使用 Trainer 自带的 DataLoader，自动携带 collator、分布式采样器、pin_memory 等设置
    dataloader = trainer.get_eval_dataloader(dataset)

    # 若是 PEFT/LoRA，关闭 adapter，用基座权重前向
    base_model_ctx = getattr(trainer.accelerator.unwrap_model(model), "disable_adapter", None)
    base_ctx = base_model_ctx() if base_model_ctx is not None else _nullctx()

    trainer.ref_sentence_ce = {}
    total_seen = 0

    # 只在主进程显示进度条
    show_bar = trainer.is_world_process_zero()
    pbar = None
    total_examples = None
    if show_bar:
        try:
            total_examples = len(dataset)
        except Exception:
            total_examples = None

        if total_examples is not None:
            pbar = tqdm(total=total_examples, dynamic_ncols=True, unit="ex", desc="[TTL] Precompute CE", leave=True)
        else:
            pbar = tqdm(dynamic_ncols=True, unit="batch", desc="[TTL] Precompute CE", leave=True)

    with base_ctx, torch.no_grad():
        for batch in dataloader:
            # 与训练一致：张量放到当前设备
            input_ids = batch["input_ids"].to(model.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs["logits"]

            # 与训练一致：用 attention_mask 屏蔽 pad，并屏蔽句首
            labels_eff = input_ids.clone()
            if attn is not None:
                labels_eff = labels_eff.masked_fill(attn == 0, IGNORE_INDEX)
            labels_eff[:, 0] = IGNORE_INDEX

            ce = trainer._cal_ce(logits, labels_eff)  # [B]

            ex_ids = batch["example_id"]
            if isinstance(ex_ids, torch.Tensor):
                ex_ids = ex_ids.tolist()

            for eid, val in zip(ex_ids, ce.detach().cpu().tolist()):
                trainer.ref_sentence_ce[int(eid)] = float(val)

            # 进度条更新
            batch_size_now = len(ex_ids)
            total_seen += batch_size_now
            if pbar is not None:
                if total_examples is not None:
                    left = max(total_examples - total_seen, 0)
                    pbar.update(batch_size_now)
                    pbar.set_postfix({"seen": total_seen, "left": left, "bs": batch_size_now})
                else:
                    pbar.update(1)
                    pbar.set_postfix({"seen": total_seen, "bs": batch_size_now})

    if pbar is not None:
        pbar.close()

    if log_path is not None and trainer.is_world_process_zero():
        trainer._log_path = log_path
        with open(trainer._log_path, "a", encoding="utf-8") as f:
            print(f"[TTL] Precomputed reference CE for {total_seen} samples.", file=f)

    model.train()


def run_ttlu(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # tokenizer 与模板
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 数据集（TTL 阶段）
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="ttl",
        **tokenizer_module,
    )

    # 训练/评估集，附加 example_id
    train_dataset = dataset_module.get("train_dataset")
    eval_dataset = dataset_module.get("eval_dataset") or train_dataset
    train_dataset = _add_example_id_column(train_dataset)
    eval_dataset = _add_example_id_column(eval_dataset)

    # 模型
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # collator：右填充训练；并用包装器保证 example_id 进入 batch
    tokenizer.padding_side = "right"
    training_args.remove_unused_columns = False  # 防止删掉 example_id
    base_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        # model=model,  # 与官方对齐，这里不强行注入 model
        # pad_to_multiple_of=8 if training_args.do_train else None,
        pad_to_multiple_of=None,
        label_pad_token_id=(IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    data_collator = _wrap_collator_with_ids(base_collator)

    # 初始化 Trainer
    trainer = TTLTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 若需要，预计算参考 CE（一次性，用 base 模型；无需维护两份权重）
    if getattr(finetuning_args, "ttl_ref_mode", "precompute") == "precompute":
        log_path = os.path.join(training_args.output_dir, "ttl_log.txt")
        _precompute_reference_ce(
            trainer=trainer,
            dataset=train_dataset,
            batch_size=int(getattr(finetuning_args, "ttl_ref_batch_size", 64)),
            log_path=log_path,
        )

    # 是否在 TTL 阶段直接做推理
    direct_infer: bool = bool(getattr(finetuning_args, "ttl_direct_inference", False))

    # 两种总体流程
    setting = getattr(finetuning_args, "ttl_setting", "offline_ttl").lower()
    if setting not in {"offline_ttl", "online_ttl"}:
        raise ValueError(f"Unsupported ttl_setting: {setting}")

    # 离线：先训后（可选）推
    if setting == "offline_ttl":
        if training_args.do_train:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.save_model()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss"])

        if direct_infer:
            # 推理输出目录
            pred_out = os.path.join(
                training_args.output_dir,
                f"predict-temperature_{generating_args.temperature}-max_new_tokens_{generating_args.max_new_tokens}",
            )
            _predict_and_save_jsonl(trainer, eval_dataset, tokenizer, pred_out, generating_args)

    # 在线：分片「先推再训」或「只训」，每片各自保存 LoRA
    else:
        bs = int(getattr(finetuning_args, "ttl_streaming_batch_size", 100))
        n = len(train_dataset)
        num_batches = n // bs + (1 if n % bs != 0 else 0)

        base_out = training_args.output_dir
        for k in range(num_batches):
            start, end = k * bs, min((k + 1) * bs, n)
            logger.info_rank0(f"Processing streaming batch {k + 1}/{num_batches}: [{start}, {end})")

            sub_train = train_dataset.select(range(start, end))
            sub_eval = eval_dataset.select(range(start, end))

            # 重新绑定当前子数据集
            trainer.train_dataset = sub_train
            trainer.eval_dataset = sub_eval

            # 子片输出目录
            sub_out = os.path.join(base_out, f"online_step_{k:04d}")
            trainer.args.output_dir = sub_out
            os.makedirs(sub_out, exist_ok=True)

            # 需要推理则先推
            if direct_infer:
                pred_out = os.path.join(
                    sub_out,
                    f"predict-temperature_{generating_args.temperature}-max_new_tokens_{generating_args.max_new_tokens}",
                )
                _predict_and_save_jsonl(trainer, sub_eval, tokenizer, pred_out, generating_args)

            # 训练该子片
            train_result = trainer.train(resume_from_checkpoint=None)
            trainer.save_model()  # 保存该子片对应的 LoRA/adapter
            trainer.log_metrics(f"train_stream_{k}", train_result.metrics)
            trainer.save_metrics(f"train_stream_{k}", train_result.metrics)
            trainer.save_state()

        # 训练结束把输出目录恢复
        trainer.args.output_dir = base_out

    # 训练或评估结束：常规评估（如果打开）
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 卡片与推送
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
