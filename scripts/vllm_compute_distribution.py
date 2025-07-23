# scripts/vllm_compute_entropy.py
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

"""
与原版 scripts/vllm_compute_distribution.py 的差异：
1. SamplingParams.logprobs → 200，engine_args["max_logprobs"] 同步 200。
2. 每个生成 token 仅计算基于 top‑200 的 Shannon 熵（自然对数单位）。
3. 保存字段精简为：prompt / generated / entropies (List[float32])。
4. 仍以 Parquet(Zstd) 流式追加写入，保持低内存占用并附带 tqdm 进度条。
"""

import gc
import math
import os
from typing import List, Optional

import fire
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def _shannon_entropy(step_logprobs) -> float:
    """给定 step_logprobs (Dict[token_id, LogProbInfo]) 计算熵 (nats)。"""
    # 提取对数概率并转为概率
    probs: List[float] = [math.exp(lp.logprob) for lp in step_logprobs.values()]
    Z = math.fsum(probs)
    if Z == 0.0:
        return 0.0
    inv_Z = 1.0 / Z
    ent = 0.0
    for p in probs:
        p *= inv_Z
        ent -= p * math.log(p + 1e-40)
    return ent


def get_token_metrics(
    model_name_or_path: str,
    output_dir: str = "output_metrics",
    adapter_name_or_path: Optional[str] = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
    gpu_memory_utilization: float = 0.92,
):
    """批量推理并记录每个生成 token 的信息熵 (top‑200)。结果流式写 Parquet。"""

    if pipeline_parallel_size > get_device_count():
        raise ValueError("流水线并行数应小于可用 GPU 的数量。")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_logprobs": 200,  # 与 SamplingParams.logprobs 对齐
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}
    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)

    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "sft", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,
        top_k=generating_args.top_k or -1,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
        logprobs=200,  # 关键：请求 top‑200 logprobs
        min_tokens=10,
    )

    lora_request = None
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])

    # --- 准备 Parquet Writer ---
    os.makedirs(output_dir, exist_ok=True)
    sanitized_model = model_name_or_path.replace("/", "_")
    sanitized_dataset = dataset.replace("/", "_").replace(":", "_")
    metrics_path = os.path.join(output_dir, f"{sanitized_model}_{sanitized_dataset}_entropy.parquet")

    schema = pa.schema(
        [
            ("prompt", pa.string()),
            ("generated", pa.string()),
            ("entropies", pa.list_(pa.float32())),
        ]
    )
    writer: Optional[pq.ParquetWriter] = None

    total = len(train_dataset) if max_samples is None else min(len(train_dataset), max_samples)
    for start in tqdm(range(0, total, batch_size), desc=f"Inferring {sanitized_model}"):
        end = min(start + batch_size, total)
        if hasattr(train_dataset, "select"):
            batch = train_dataset.select(range(start, end))
        else:
            subset = [train_dataset[k] for k in range(start, end)]
            batch = {k: [d[k] for d in subset] for k in subset[0]}

        vllm_inputs, prompts = [], []
        for input_ids in batch["input_ids"]:
            vllm_inputs.append({"prompt_token_ids": input_ids, "multi_modal_data": None})
            prompts.append(tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens))

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

        batch_metrics = []
        for idx, result in enumerate(results):
            out = result.outputs[0]
            raw_logprobs = out.logprobs or []

            # 计算每个生成 token 的熵
            entropies: List[float] = []
            for step_lp in raw_logprobs:
                entropies.append(_shannon_entropy(step_lp))

            batch_metrics.append(
                {
                    "prompt": prompts[idx],
                    "generated": out.text,
                    "entropies": entropies,
                }
            )

            if max_samples and (start + idx + 1) >= max_samples:
                break

        # 写入磁盘
        table = pa.Table.from_pylist(batch_metrics, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(metrics_path, schema, compression="zstd")
        writer.write_table(table)

        # 内存清理
        del batch_metrics, table, results
        torch.cuda.empty_cache()
        gc.collect()

        if max_samples and (end) >= max_samples:
            break

    if writer is not None:
        writer.close()
        print(f"Saved entropies to {metrics_path}")


if __name__ == "__main__":
    fire.Fire(get_token_metrics)
