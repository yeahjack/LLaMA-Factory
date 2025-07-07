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
import re
import time
from statistics import mean
from typing import List, Optional

import fire
from datasets import load_dataset
from tqdm import tqdm

try:
    import evaluate
    import torch

    # Defer loading to when they are actually needed
except ImportError:
    print(
        "请安装所需依赖: pip install evaluate bert_score sacrebleu rouge_score transformers torch"
    )
    raise


def _extract_answer(text: str, dataset_type: str) -> Optional[str]:
    """
    根据数据集类型从单个文本字符串中提取答案的辅助函数。
    此版本针对 metamathqa 和 gsm8k 进行了优化。
    """
    if not isinstance(text, str) or not text.strip():
        return None

    text = text.strip()

    if dataset_type == "logiqa":
        match = re.search(r"[Aa]nswer:\s*([A-D])", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(r"\b([A-D])\b\s*$", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    if dataset_type in ["gsm8k", "metamathqa"]:
        last_marker_pos = text.rfind('####')
        if last_marker_pos != -1:
            answer_text = text[last_marker_pos + 4:]
            match = re.search(r"-?[\d,]+\.?\d*|-?\d+\.?[\d,]*", answer_text)
            if match:
                return match.group(0).replace(",", "").strip()

        matches = re.findall(r"\\boxed{([^}]+)}", text)
        if matches:
            answer_text = matches[-1]
            match = re.search(r"-?[\d,]+\.?\d*|-?\d+\.?[\d,]*", answer_text)
            if match:
                return match.group(0).replace(",", "").strip()
            return answer_text.strip()

        matches = re.findall(
            r"(?:[Tt]he answer is|[Aa]nswer:)\s*\$?(-?[\d,./]+)", text)
        if matches:
            return matches[-1].replace(",", "").replace("$", "").strip()

        if last_marker_pos == -1 and not matches:
            match = re.search(r"(-?[\d,./]+)\s*$", text)
            if match:
                return match.group(1).replace(",", "").strip()

        return None

    return text.strip()


def calculate_exact_match(predictions: List[str], references: List[str],
                          dataset_type: str) -> List[float]:
    """
    根据数据集类型，使用更鲁棒的逻辑计算精确匹配（Exact Match）分数。
    """
    em_scores = []
    for pred_str, label_str in tqdm(
            zip(predictions, references),
            desc=f"Calculating Exact Match for '{dataset_type}'",
            total=len(predictions),
    ):
        em_score = 0.0
        pred_ans = _extract_answer(pred_str, dataset_type)
        label_ans = _extract_answer(label_str, dataset_type)

        if pred_ans is not None and label_ans is not None:
            try:
                if float(pred_ans) == float(label_ans):
                    em_score = 1.0
            except (ValueError, TypeError):
                if pred_ans.strip().lower() == label_ans.strip().lower():
                    em_score = 1.0

        em_scores.append(em_score)

    return em_scores


def main(filename: str, output_filename: str, *, metrics="all"):
    """主函数，加载数据、计算并保存评估结果。.

    Args:
        filename (str): 输入的JSON文件名 (包含预测和标签)。
        output_filename (str): 输出评估结果的JSON文件名。
        metrics (str or tuple): 指定要计算的指标。
                       可以是用逗号分隔的字符串 "rouge,em"
                       也可以是多个参数 --metrics rouge em
                       可选值: "bleu", "rouge", "bertscore" (或"bs"), "em" (或"exact_match")。
                       默认为 "all"，计算所有指标。
    """
    start_time = time.time()

    # --- 步骤 1: 用更健壮的方式解析指标参数 ---
    metric_list = []
    if isinstance(metrics, str):
        metric_list = [m.strip().lower() for m in metrics.split(',')]
    elif isinstance(metrics, (list, tuple)):
        metric_list = [str(m).strip().lower() for m in metrics]

    metric_map = {
        "bs": "bertscore",
        "bertscore": "bertscore",
        "rouge": "rouge",
        "bleu": "bleu",
        "em": "em",
        "exact_match": "em"
    }

    if "all" in metric_list:
        enabled_metrics = {"bleu", "rouge", "bertscore", "em"}
    else:
        enabled_metrics = {
            metric_map[m]
            for m in metric_list if m in metric_map
        }

    print(f"将要计算的指标: {', '.join(sorted(list(enabled_metrics)))}")

    print("\n正在加载数据集...")
    try:
        dataset = load_dataset("json", data_files=filename, split="train")
    except Exception as e:
        print(f"无法加载文件 {filename}。请检查文件路径和格式。错误: {e}")
        return

    predictions = [sample["predict"] for sample in dataset]
    references = [sample["label"] for sample in dataset]

    average_score = {}

    # --- 步骤 2: 按需计算各项指标 ---
    if "bertscore" in enabled_metrics:
        print("\n正在批量计算 BERTScore (可能使用GPU)...")
        bertscore = evaluate.load("bertscore")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"BERTScore 将在设备上运行: {device}")
        bertscore_results = bertscore.compute(predictions=predictions,
                                              references=references,
                                              lang="en",
                                              model_type="bert-base-uncased",
                                              device=device,
                                              batch_size=32,
                                              verbose=True)
        average_score["bertscore-f1"] = round(
            mean(bertscore_results["f1"]) * 100, 4)
        print("BERTScore 计算完成。")

    if "rouge" in enabled_metrics:
        print("\n正在批量计算 ROUGE...")
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=predictions,
                                     references=references)
        average_score["rouge-1"] = round(rouge_scores["rouge1"] * 100, 4)
        average_score["rouge-2"] = round(rouge_scores["rouge2"] * 100, 4)
        average_score["rouge-l"] = round(rouge_scores["rougeL"] * 100, 4)
        average_score['rouge-Lsum'] = round(rouge_scores["rougeLsum"] * 100, 4)
        print("ROUGE 计算完成。")

    if "bleu" in enabled_metrics:
        print("\n正在批量计算 BLEU...")
        bleu = evaluate.load("bleu")
        bleu_scores = bleu.compute(predictions=predictions,
                                   references=[[ref] for ref in references])
        average_score["bleu"] = round(bleu_scores["bleu"] * 100, 4)
        print("BLEU 计算完成。")

    if "em" in enabled_metrics:
        fn_lower = filename.lower()
        dataset_type = "default"
        if "logiqa" in fn_lower:
            dataset_type = "logiqa"
        elif "gsm8k" in fn_lower:
            dataset_type = "gsm8k"
        elif "metamath" in fn_lower:
            dataset_type = "metamathqa"

        print(f"\n检测到数据集类型为: '{dataset_type}'，将为精确匹配启用特定提取逻辑。")
        em_scores = calculate_exact_match(predictions, references,
                                          dataset_type)
        average_score["exact_match"] = round(mean(em_scores) * 100, 4)
        print("精确匹配计算完成。")

    # --- 步骤 3: 打印并保存最终结果 ---
    print("\n--- 最终评估结果 ---")
    if not average_score:
        print("没有计算任何指标。请检查 'metrics' 参数。")
    else:
        sorted_tasks = sorted(average_score.keys())
        for task in sorted_tasks:
            print(f"{task:<15}: {average_score[task]:.4f}")

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(average_score, f, indent=4)
        print(f"\n分数文件已保存至: {output_filename}")

    print(f"\n计算完成，总用时 {time.time() - start_time:.3f} 秒。")


if __name__ == "__main__":
    fire.Fire(main)
