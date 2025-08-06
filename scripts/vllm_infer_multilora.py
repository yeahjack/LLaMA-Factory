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
import os
from typing import Optional, List

import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_paths: List[str] = None,
    datasets: List[str] = ["alpaca_en_demo"],
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_prefix: str = "generated_predictions",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
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
    gpu_memory_utilization: float = 0.90,
):
    r"""Perform batch generation using vLLM engine with multiple LoRA adapters and datasets.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --datasets ["dataset1","dataset2"] --adapter_name_or_paths ["adapter1","adapter2"]
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    # Ensure adapter_name_or_paths is None or has same length as datasets
    if adapter_name_or_paths is not None and len(adapter_name_or_paths) != len(datasets):
        raise ValueError("Number of adapters must match number of datasets")

    # Create base model args
    base_model_args = dict(
        model_name_or_path=model_name_or_path,
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

    # Load tokenizer and template once
    model_args_temp = get_infer_args({**base_model_args, "dataset": datasets[0]})[0]
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args_temp)
    tokenizer = tokenizer_module["tokenizer"]
    
    # Initialize vLLM engine
    engine_args = {
        "model": model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args_temp.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": adapter_name_or_paths is not None,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    
    if isinstance(model_args_temp.vllm_config, dict):
        engine_args.update(model_args_temp.vllm_config)

    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        repetition_penalty=repetition_penalty or 1.0,
        temperature=temperature,
        top_p=top_p or 1.0,
        top_k=top_k or -1,
        stop_token_ids=None,  # Will be set per dataset
        max_tokens=max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
        min_tokens=20
    )

    # Process each dataset with its corresponding adapter
    for idx, dataset_name in enumerate(datasets):
        adapter_path = adapter_name_or_paths[idx] if adapter_name_or_paths else None
        
        print(f"\n{'='*70}")
        print(f"Processing dataset: {dataset_name}")
        if adapter_path:
            print(f"Using adapter: {adapter_path}")
        print(f"{'='*70}\n")

        # Get model args for current dataset and adapter
        current_args = {**base_model_args, "dataset": dataset_name}
        if adapter_path:
            current_args["adapter_name_or_path"] = adapter_path
        
        model_args, data_args, _, generating_args = get_infer_args(current_args)
        
        # Get template for current dataset
        template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
        template_obj.mm_plugin.expand_mm_tokens = False
        
        # Update sampling params with dataset-specific stop tokens
        sampling_params.stop_token_ids = template_obj.get_stop_token_ids(tokenizer)
        
        # Update engine args for multimodal if needed
        if template_obj.mm_plugin.__class__.__name__ != "BasePlugin" and "limit_mm_per_prompt" not in engine_args:
            engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}
        
        # Load dataset
        dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
        train_dataset = dataset_module["train_dataset"]

        # Create LoRA request if adapter is provided
        lora_request = LoRARequest(f"adapter_{idx}", idx + 1, adapter_path) if adapter_path else None

        # Store results for current dataset
        all_prompts, all_preds, all_labels = [], [], []

        # Batch process
        for i in tqdm(range(0, len(train_dataset), batch_size), desc=f"Processing {dataset_name}"):
            vllm_inputs, prompts, labels = [], [], []
            batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

            for j in range(len(batch["input_ids"])):
                if batch["images"][j] is not None:
                    image = batch["images"][j]
                    multi_modal_data = {
                        "image": template_obj.mm_plugin._regularize_images(
                            image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                        )["images"]
                    }
                elif batch["videos"][j] is not None:
                    video = batch["videos"][j]
                    multi_modal_data = {
                        "video": template_obj.mm_plugin._regularize_videos(
                            video,
                            image_max_pixels=image_max_pixels,
                            image_min_pixels=image_min_pixels,
                            video_fps=video_fps,
                            video_maxlen=video_maxlen,
                        )["videos"]
                    }
                elif batch["audios"][j] is not None:
                    audio = batch["audios"][j]
                    audio_data = template_obj.mm_plugin._regularize_audios(
                        audio,
                        sampling_rate=16000,
                    )
                    multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
                else:
                    multi_modal_data = None

                vllm_inputs.append({"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data})
                prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
                labels.append(
                    tokenizer.decode(
                        list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                        skip_special_tokens=skip_special_tokens,
                    )
                )

            results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
            preds = [result.outputs[0].text for result in results]

            all_prompts.extend(prompts)
            all_preds.extend(preds)
            all_labels.extend(labels)
            gc.collect()

        # Save results for current dataset with proper directory structure
        # Extract method info from adapter path if available
        if adapter_path:
            # adapter_path format: saves/llama31_8b/lora/{method}_search{suffix}/{dataset}_{gen_len}
            path_parts = adapter_path.split('/')
            if len(path_parts) >= 2:
                method_dir = path_parts[-2]  # e.g., "ppl_ppl_search_MA_seq_nokl"
                dataset_dir = path_parts[-1]  # e.g., "db_wealth_0"
                gen_len = dataset_dir.split('_')[-1]  # extract generation length
                
                # Create output directory matching original structure
                output_dir = f"{save_prefix}/{method_dir.replace('_search', '_' + gen_len)}"
                os.makedirs(output_dir, exist_ok=True)
                save_name = f"{output_dir}/{dataset_name.replace('_test', '')}.jsonl"
            else:
                # Fallback to simple naming
                save_name = f"{save_prefix}_{dataset_name}.jsonl"
        else:
            # No adapter case (base model only)
            save_name = f"{save_prefix}_{dataset_name}.jsonl"
        
        with open(save_name, "w", encoding="utf-8") as f:
            for text, pred, label in zip(all_prompts, all_preds, all_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

        print(f"\n{len(all_prompts)} results saved to {save_name}")

    print("\n" + "*" * 70)
    print(f"All inference tasks completed!")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)