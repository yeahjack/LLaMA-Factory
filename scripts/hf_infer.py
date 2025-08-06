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

import gc
import json
from typing import Optional

import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

# 导入 PEFT 相关库用于 LoRA
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


def hf_infer(
    model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    batch_size: int = 4,
    # 多模态参数保留，但在当前脚本中未直接用于批处理
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
):
    r"""
    使用 Hugging Face Transformers 的 generate 方法进行批量生成。
    """
    if adapter_name_or_path is not None and PeftModel is None:
        raise ValueError("PEFT is not installed. Please install it via `pip install peft`.")

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
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    if seed is not None:
        torch.manual_seed(seed)

    # 加载 Tokenizer 和 Template
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # 对于批处理生成，padding在左边是必须的
    tokenizer.padding_side = 'left'
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 加载模型
    print("Loading model...")
    quantization_config = None
    if model_args.quantization_bit == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif model_args.quantization_bit == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.infer_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    if model_args.adapter_name_or_path is not None:
        print(f"Loading and merging LoRA adapter from {model_args.adapter_name_or_path[0]}")
        model = PeftModel.from_pretrained(model, model_args.adapter_name_or_path[0])
        model = model.merge_and_unload()
    
    model.eval()

    # 优化 3: 使用 torch.compile 加速模型
    try:
        model = torch.compile(model)
        print("Model compiled successfully with torch.compile.")
    except Exception as e:
        print(f"Failed to compile model with torch.compile: {e}")

    print("Model loaded and prepared.")

    # 加载数据集
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]
    if max_samples is not None and len(train_dataset) > max_samples:
        train_dataset = train_dataset.select(range(max_samples))

    all_prompts, all_preds, all_labels = [], [], []

    # 按批次进行推理
    for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference"):
        batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

        # ** FIX: Removed `truncation` and `max_length` from `tokenizer.pad` **
        # The `get_dataset` function already handles truncation.
        inputs = tokenizer.pad(
            {"input_ids": batch["input_ids"]},
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # 解码操作仅用于最后保存结果
        current_prompts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["input_ids"]]
        current_labels = [
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, ids)), skip_special_tokens=True)
            for ids in batch["labels"]
        ]
        
        # 精简贪婪解码参数
        gen_kwargs = {
            "max_new_tokens": generating_args.max_new_tokens,
            "repetition_penalty": generating_args.repetition_penalty or 1.0,
            "do_sample": False,  # 明确指定贪婪解码
            "eos_token_id": template_obj.get_stop_token_ids(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        preds = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        all_prompts.extend(current_prompts)
        all_preds.extend(preds)
        all_labels.extend(current_labels)
        

    # 在循环外一次性写入所有结果
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(hf_infer)