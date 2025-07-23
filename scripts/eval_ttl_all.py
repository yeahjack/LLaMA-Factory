#!/usr/bin/env python
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
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean
from typing import List, Optional, Tuple, Set, Union

import fire
from datasets import load_dataset
from tqdm import tqdm

# 全局变量，用于在子进程中延迟加载库
evaluate = None
torch = None


# ===================================================================================
# ==               模块一: 单个文件的评估逻辑 (由工作进程调用)                     ==
# ===================================================================================

def _lazy_import():
    """延迟导入cuda敏感库，确保在设置CUDA_VISIBLE_DEVICES后执行。"""
    global evaluate, torch
    if evaluate is None:
        import evaluate as _evaluate
        evaluate = _evaluate
    if torch is None:
        import torch as _torch
        torch = _torch


def _extract_answer(text: str, dataset_type: str) -> Optional[str]:
    """根据数据集类型从单个文本字符串中提取答案的辅助函数。"""
    if not isinstance(text, str) or not text.strip(): return None
    text = text.strip()
    if dataset_type == "logiqa":
        match = re.search(r"[Aa]nswer:\s*([A-D])", text, re.IGNORECASE) or re.search(r"\b([A-D])\b\s*$", text, re.IGNORECASE)
        return match.group(1).upper() if match else None
    if dataset_type in ["gsm8k", "metamathqa"]:
        if (last_marker_pos := text.rfind('####')) != -1:
            answer_text = text[last_marker_pos + 4:]
            if (match := re.search(r"-?[\d,]+\.?\d*|-?\d+\.?[\d,]*", answer_text)): return match.group(0).replace(",", "").strip()
        if (matches := re.findall(r"\\boxed{([^}]+)}", text)):
            answer_text = matches[-1]
            if (match := re.search(r"-?[\d,]+\.?\d*|-?\d+\.?[\d,]*", answer_text)): return match.group(0).replace(",", "").strip()
            return answer_text.strip()
        if (matches := re.findall(r"(?:[Tt]he answer is|[Aa]nswer:)\s*\$?(-?[\d,./]+)", text)): return matches[-1].replace(",", "").replace("$", "").strip()
        if not matches and last_marker_pos == -1:
            if (match := re.search(r"(-?[\d,./]+)\s*$", text)): return match.group(1).replace(",", "").strip()
        return None
    return text.strip()


def calculate_exact_match(predictions: List[str], references: List[str], dataset_type: str) -> List[float]:
    """计算精确匹配分数，带tqdm进度条。"""
    em_scores = []
    for pred_str, label_str in tqdm(zip(predictions, references), desc=f"  EM for '{dataset_type}'", total=len(predictions), leave=False, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        em_score = 0.0
        pred_ans, label_ans = _extract_answer(pred_str, dataset_type), _extract_answer(label_str, dataset_type)
        if pred_ans is not None and label_ans is not None:
            try:
                if abs(float(pred_ans) - float(label_ans)) < 1e-6: em_score = 1.0
            except (ValueError, TypeError):
                if pred_ans.strip().lower() == label_ans.strip().lower(): em_score = 1.0
        em_scores.append(em_score)
    return em_scores


def run_single_evaluation(filename: str, output_filename: str, enabled_metrics: Set[str]) -> dict:
    """对单个文件进行评估，支持创建或更新(追加)模式。"""
    _lazy_import()
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f: average_score = json.load(f)
        except json.JSONDecodeError:
            return {"status": "failed", "reason": f"Corrupted JSON in existing metrics file: {os.path.basename(output_filename)}"}
    else:
        average_score = {}

    try:
        dataset = list(load_dataset("json", data_files=filename, split="train"))
    except Exception as e:
        return {"status": "failed", "reason": f"Failed to load dataset: {e}", "traceback": traceback.format_exc()}
    
    if not dataset: return {"status": "skipped", "reason": "empty file"}

    predictions = [sample.get("predict") for sample in dataset]
    references = [sample.get("label") for sample in dataset]
    if None in predictions or None in references:
        return {"status": "skipped", "reason": "missing 'predict' or 'label' field"}

    if "bertscore" in enabled_metrics:
        bertscore = evaluate.load("bertscore", keep_in_memory=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="bert-base-uncased", device=device, batch_size=32, verbose=False)
        average_score["bertscore-f1"] = round(mean(bertscore_results["f1"]) * 100, 4)

    if "rouge" in enabled_metrics:
        rouge = evaluate.load("rouge", keep_in_memory=True)
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        average_score.update({"rouge-1": round(rouge_scores["rouge1"] * 100, 4), "rouge-2": round(rouge_scores["rouge2"] * 100, 4), "rouge-l": round(rouge_scores["rougeL"] * 100, 4), "rouge-Lsum": round(rouge_scores["rougeLsum"] * 100, 4)})

    if "bleu" in enabled_metrics:
        bleu = evaluate.load("bleu", keep_in_memory=True)
        bleu_scores = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
        average_score["bleu"] = round(bleu_scores["bleu"] * 100, 4)

    if "em" in enabled_metrics:
        fn_lower, dataset_type = filename.lower(), "default"
        if "logiqa" in fn_lower: dataset_type = "logiqa"
        elif "gsm8k" in fn_lower: dataset_type = "gsm8k"
        elif "metamath" in fn_lower: dataset_type = "metamathqa"
        em_scores = calculate_exact_match(predictions, references, dataset_type)
        average_score["exact_match"] = round(mean(em_scores) * 100, 4)

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4, ensure_ascii=False)
    
    return {"status": "success", "scores": average_score}


# ===================================================================================
# ==               模块二: 工作进程封装 (由主进程创建)                             ==
# ===================================================================================

def worker_evaluate(task_args: Tuple[str, str, int, str, Set[str]]) -> dict:
    """工作进程执行的函数。"""
    input_path, output_path, gpu_id, root_dir, enabled_metrics = task_args
    if gpu_id != -1: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start_time = time.time()
    relative_path = os.path.relpath(input_path, root_dir)
    try:
        result = run_single_evaluation(input_path, output_path, enabled_metrics)
    except Exception as e:
        result = {"status": "failed", "reason": "Unhandled exception in worker", "error": str(e), "traceback": traceback.format_exc()}
    result['elapsed'] = time.time() - start_time
    result['path'] = relative_path
    return result


# ===================================================================================
# ==               模块三: 主进程与任务编排 (脚本入口)                             ==
# ===================================================================================

def process_all_results(
    root_dir: str = "/home/yijiexu/LLaMA-Factory/results",
    num_workers: int = 12,
    gpus: Union[str, Tuple[str]] = "0,1,2,3,4,5",
    metric: str = "all",
    force: bool = False
):
    """
    并行评估所有结果文件，支持计算全部指标或追加单个指标。

    Args:
        root_dir (str): 实验结果的根目录。
        num_workers (int): 并行处理文件的进程数。
        gpus (Union[str, Tuple[str]]): 以逗号分隔的GPU ID列表。
        metric (str): 要计算的指标。可选: "all", "bertscore"。
        force (bool): 如果为True，即使指标已存在也强制重新计算。
    """
    print(f"--- 开始并行批量评估 (模式: {metric}) ---")
    
    # *** 关键修复：健壮地处理 gpus 参数 ***
    gpu_ids = []
    if gpus:
        gpus_str = ""
        if isinstance(gpus, tuple):
            gpus_str = ",".join(map(str, gpus))
        elif isinstance(gpus, (str, int)):
            gpus_str = str(gpus)

        if gpus_str:
            try:
                gpu_ids = [int(g.strip()) for g in gpus_str.split(',')]
            except ValueError:
                print(f"错误: 'gpus' 参数格式不正确。收到了 '{gpus}'。应为逗号分隔的整数，例如 '0,1,2'。")
                return
            
    print(f"配置: 根目录='{root_dir}', 工作进程数={num_workers}, 可用GPU={gpu_ids if gpu_ids else 'CPU'}, 强制重算={force}")

    if metric.lower() == "all":
        enabled_metrics = {"bleu", "rouge", "bertscore", "em"}
        metric_keys_to_check = [] 
    elif metric.lower() == "bertscore":
        enabled_metrics = {"bertscore"}
        metric_keys_to_check = ["bertscore-f1"]
    else:
        print(f"错误: 不支持的指标 '{metric}'。请选择 'all' 或 'bertscore'。"); return

    print("\n[1/3] 正在扫描文件并准备任务...")
    all_potential_files, tasks_to_run, skipped_count = [], [], 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jsonl") and not (basename.endswith("_metrics") or basename.endswith("_LLM_EM")):
                all_potential_files.append(os.path.join(dirpath, filename))
    all_potential_files.sort()
    
    for input_path in all_potential_files:
        basename, _ = os.path.splitext(input_path)
        output_path = f"{basename}_metrics.json"

        if not force:
            if not os.path.exists(output_path) and metric_keys_to_check:
                skipped_count += 1; continue
            if os.path.exists(output_path) and metric_keys_to_check:
                try:
                    with open(output_path, 'r', encoding='utf-8') as f: existing_data = json.load(f)
                    if all(key in existing_data for key in metric_keys_to_check):
                        skipped_count += 1; continue
                except (json.JSONDecodeError, FileNotFoundError): pass
            elif os.path.exists(output_path) and not metric_keys_to_check:
                skipped_count += 1; continue

        assigned_gpu = gpu_ids[len(tasks_to_run) % len(gpu_ids)] if gpu_ids else -1
        tasks_to_run.append((input_path, output_path, assigned_gpu, root_dir, enabled_metrics))
    
    if not tasks_to_run:
        print(f"扫描完成。没有需要处理的新文件。跳过 {skipped_count} 个已完成/不适用的任务。"); return
    print(f"扫描完成。找到 {len(tasks_to_run)} 个新任务。跳过 {skipped_count} 个已完成/不适用的任务。")

    print("\n[2/3] 开始并行处理任务...")
    success_count, fail_count, skipped_in_run_count = 0, 0, 0
    failed_tasks_details = []
    overall_start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_evaluate, task): task for task in tasks_to_run}
        pbar = tqdm(total=len(tasks_to_run), desc="Overall Progress", ncols=120, dynamic_ncols=True)
        for future in as_completed(futures):
            try:
                result = future.result()
                status, path_str = result['status'], result.get('path', 'unknown_file')
                if status == 'failed':
                    fail_count += 1; status_char = "❌"; failed_tasks_details.append(result)
                    pbar.write("\n" + "!"*35 + " TASK FAILED " + "!"*34)
                    pbar.write(f"  [File]: {result['path']}\n  [Reason]: {result.get('reason') or result.get('error')}")
                    if 'traceback' in result and result['traceback']: pbar.write(f"  [Traceback]:\n{result['traceback']}")
                    pbar.write("!"*80 + "\n")
                elif status == 'success': success_count += 1; status_char = "✅"
                elif status == 'skipped': skipped_in_run_count += 1; status_char = "⏩"
                pbar.set_postfix_str(f"Success: {success_count}, Failed: {fail_count}, Last: {path_str} {status_char}", refresh=True)
            except Exception as e:
                task = futures[future]
                relative_path = os.path.relpath(task[0], task[3])
                fail_count += 1; tb_str = traceback.format_exc(); failed_tasks_details.append({"status": "crashed", "path": relative_path, "error": str(e), "traceback": tb_str})
                pbar.write(f"\n" + "!"*33 + " WORKER CRASHED " + "!"*32 + f"\n  [File]: {relative_path}\n  [Error]: {e}\n  [Traceback]:\n{tb_str}\n" + "!"*80 + "\n")
                pbar.set_postfix_str(f"Success: {success_count}, Failed: {fail_count}, Last: {relative_path} 🔥CRASHED🔥", refresh=True)
            pbar.update(1)
        pbar.close()

    overall_end_time = time.time()
    print("\n\n[3/3] 所有任务处理完毕，生成摘要报告。")
    print("#"*80)
    print("###" + " " * 30 + "批量评估任务完成！" + " " * 29 + "###")
    print("#"*80)
    print(f"总计用时: {overall_end_time - overall_start_time:.2f} 秒")
    print(f"  - 成功处理/更新文件数: {success_count} ✅")
    print(f"  - 失败文件数: {fail_count} ❌")
    print(f"  - 因文件内容问题跳过的文件数: {skipped_in_run_count} ⏩")
    print(f"  - 因指标已存在/不适用而跳过的文件数: {skipped_count} ⏭️")
    if failed_tasks_details:
        print("\n--- 失败任务简要回顾 (详细日志见上方) ---")
        for i, failure in enumerate(failed_tasks_details, 1): print(f"  {i}. {failure['path']} ({failure['status']})")
    print("#"*80)


if __name__ == "__main__":
    fire.Fire(process_all_results)