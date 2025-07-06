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

import fire
from datasets import load_dataset
from tqdm import tqdm

try:
    import evaluate
    import torch

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

except ImportError:
    print(
        "请安装所需依赖: pip install evaluate bert_score sacrebleu rouge_score transformers torch"
    )
    raise


def _extract_answer(text: str, dataset_type: str) -> str | None:
    """
    根据数据集类型从单个文本字符串中提取答案的辅助函数。
    """
    if not isinstance(text, str) or not text.strip():
        return None

    text = text.strip()

    if dataset_type == "logiqa":
        # 优先匹配 "Answer: A" 这样的显式模式
        match = re.search(r"[Aa]nswer:\s*([A-D])", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 如果没有，则匹配末尾的独立字母 A, B, C, D
        # \b 是单词边界，确保不会从 "Apple" 中匹配到 "A"
        match = re.search(r"\b([A-D])\b\s*$", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        return None

    # 对于 gsm8k 和 metamathqa，使用一套共享的、有优先级的提取规则
    if dataset_type in ["gsm8k", "metamathqa"]:
        # 规则 1: 匹配 '#### <答案>'。这是最可靠的标记。
        # 使用 findall 获取所有匹配项，并取最后一个作为最终答案。
        matches = re.findall(r"####\s*\$?\s*(-?[\d,./\\]+)", text)
        if matches:
            ans = matches[-1]
            # 清理答案字符串中的常见非数值字符
            return ans.replace(",", "").replace("$", "").strip()

        # 规则 2: 匹配 LaTeX 的 '\boxed{<答案>}'
        match = re.search(r"\\boxed{([^}]+)}", text)
        if match:
            return match.group(1).strip()
            
        # 规则 3: 匹配 'the answer is <答案>' 等自然语言模式
        match = re.search(
            r"(?:[Tt]he answer is|[Aa]nswer:)\s*\$?(-?[\d,./\\]+)",
            text
        )
        if match:
            return match.group(1).replace(",", "").replace("$", "").strip()

        # 规则 4: 提取字符串末尾的数字/分数作为最后的备选方案
        match = re.search(r"(-?[\d,./]+)\s*$", text)
        if match:
            return match.group(1).replace(",", "").strip()
            
    # 默认/回退逻辑：如果以上规则都不适用，则直接返回原始字符串的清理版
    return text.strip()


def calculate_exact_match(
    predictions: list, references: list, dataset_type: str
) -> list:
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

        # 为 prediction 和 label 分别提取答案
        pred_ans = _extract_answer(pred_str, dataset_type)
        label_ans = _extract_answer(label_str, dataset_type)

        # 只有在两边都成功提取到答案时才进行比较
        if pred_ans is not None and label_ans is not None:
            # 尝试将答案作为浮点数进行比较，这能处理 "2" vs "2.0" 的情况
            try:
                if float(pred_ans) == float(label_ans):
                    em_score = 1.0
            except (ValueError, TypeError):
                # 如果转换失败（例如答案是分数 "15/56" 或字母 "C"），
                # 则回退到大小写不敏感的字符串比较
                if pred_ans.strip().lower() == label_ans.strip().lower():
                    em_score = 1.0
        
        em_scores.append(em_score)

    return em_scores


def main(filename: str, output_filename: str):
    """主函数，加载数据、计算并保存评估结果（批处理优化版）"""
    start_time = time.time()

    print("正在加载数据集...")
    try:
        dataset = load_dataset("json", data_files=filename, split="train")
    except Exception as e:
        print(f"无法加载文件 {filename}。请检查文件路径和格式。错误: {e}")
        return

    # --- 步骤1: 将所有预测和标签一次性提取到列表中 ---
    predictions = [sample["predict"] for sample in dataset]
    references = [sample["label"] for sample in dataset]

    average_score = {}

    # --- 步骤2: 批量计算GPU密集型指标 (BERTScore) ---
    print("\n正在批量计算 BERTScore (使用GPU)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"BERTScore 将在设备上运行: {device}")
    bertscore_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        device=device,
        batch_size=32,
        verbose=True,
    )
    average_score["bertscore-f1"] = round(mean(bertscore_results["f1"]) * 100, 4)
    print(f"BERTScore 计算完成。")

    # --- 步骤3: 批量计算CPU密集型指标 (ROUGE, BLEU) ---
    print("\n正在批量计算 ROUGE...")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    average_score["rouge-1"] = round(rouge_scores["rouge1"] * 100, 4)
    average_score["rouge-2"] = round(rouge_scores["rouge2"] * 100, 4)
    average_score["rouge-l"] = round(rouge_scores["rougeL"] * 100, 4)
    average_score["rouge-lsum"] = round(rouge_scores["rougeLsum"] * 100, 4)
    print("ROUGE 计算完成。")

    print("\n正在批量计算 BLEU...")
    # BLEU 的 references 需要是列表的列表
    bleu_scores = bleu.compute(
        predictions=predictions, references=[[ref] for ref in references]
    )
    average_score["bleu-1"] = round(bleu_scores["bleu"] * 100, 1)
    average_score["bleu-2"] = round(bleu_scores["bleu"] * 100, 2)
    average_score["bleu-3"] = round(bleu_scores["bleu"] * 100, 3)
    average_score["bleu-4"] = round(bleu_scores["bleu"] * 100, 4)
    print("BLEU 计算完成。")

    # --- 步骤4: 根据文件名确定数据集类型并计算精确匹配 (EM) ---
    fn_lower = filename.lower()
    dataset_type = "default"  # 默认类型
    if "logiqa" in fn_lower:
        dataset_type = "logiqa"
    elif "gsm8k" in fn_lower:
        dataset_type = "gsm8k"
    elif "metamathqa" in fn_lower:
        dataset_type = "metamathqa"
    
    print(f"\n检测到数据集类型为: '{dataset_type}'，将为精确匹配启用特定提取逻辑。")

    em_scores = calculate_exact_match(predictions, references, dataset_type)
    average_score["exact_match"] = round(mean(em_scores) * 100, 4)
    print("精确匹配计算完成。")

    # --- 步骤5: 打印并保存最终结果 ---
    print("\n--- 评估结果 (与论文对齐) ---")
    sorted_tasks = sorted(average_score.keys())
    for task in sorted_tasks:
        print(f"{task:<15}: {average_score[task]:.4f}")

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4)

    print(f"\n计算完成，总用时 {time.time() - start_time:.3f} 秒。")
    print(f"分数文件已保存至: {output_filename}")


if __name__ == "__main__":
    fire.Fire(main)