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

"""本脚本在你原始评测代码的基础上做了如下对齐与增强：.

1) 指标计算逻辑与“官方实现”保持一致：
   - BERTScore：使用 bert_score 库与 "bert-base-multilingual-cased" 模型；在打分前对候选与参考分别做
     500 个 token 截断，并将字符串裁成等长后再计算；batch_size=16；设备优先 cuda:0 其次 cpu。
   - ROUGE：使用 rouge_score.RougeScorer，候选摘要先按句子切分并用换行连接；use_stemmer=True，
     split_summaries=True；计算 rouge1 / rouge2 / rougeL / rougeLsum 的 F1 并对样本平均。
   - BLEU：使用 NLTK 的 sentence_bleu，并使用 SmoothingFunction().method4 做平滑。
   - BLEURT（可选）：使用 bleurt-base-128；TensorFlow 启用显存按需增长；会在 ~/.bleurt/bleurt-base-128、
     当前目录和 BLEURT_CACHE 环境变量指定的目录中查找 checkpoint。
   - Accuracy / Exact Match：使用与官方 eval_utils.py 一致的抽取逻辑：
       · LogiQA：从输出中抽取 A/B/C/D；
       · GSM8K：从输出末尾抽取最后一个数值 token（做必要清洗与规整）；
       · MetaMathQA：优先抽取 \boxed{...} 或最后一个 \frac{d}{d}，并包含对带文本标签的“反向匹配”。
     以上均对提取到的字符串做必要的 LaTeX 标点与单位清理，保证与标注口径一致。

2) 输入与输出格式：
   - 输入：使用 datasets.load_dataset("json", ...) 读取，字段名期望为 "predict" 与 "label"。
   - 输出：保持为你原始 JSON 键：
       "bertscore-f1", "rouge-1", "rouge-2", "rouge-l", "rouge-Lsum", "bleu", "exact_match"
       （可选再加 "bleurt" 当且仅当显式请求 --metrics bleurt）。
   - 数值范围：本版本按照你的要求，**最终写入与打印时统一乘以 100（百分制）**，并统一四舍五入到 4 位小数。

3) metrics 选择：
   - "all" 对齐官方：仅计算 bertscore / rouge / bleu / em（不包含 bleurt）。
   - 可通过 --metrics bleurt 单独添加 BLEURT。
   - 支持别名："bs"->"bertscore", "em"/"exact_match"->"em"。

4) 类型标注遵循你的偏好：当使用类型标注时，使用 Python 内置小写类型名 list、dict、tuple 等。

依赖：
  pip install bert-score rouge-score nltk bleurt==0.0.2 tensorflow torch datasets fire tqdm

首次运行如果缺少 NLTK 资源，将自动尝试下载 punkt 与 punkt_tab。
"""

import json
import os
import re
import time
from statistics import mean
from typing import Optional

import fire
from datasets import load_dataset
from tqdm import tqdm

# =========================
# 官方口径的 EM 抽取工具（精简并对齐 scripts/eval_tlm_official/eval_utils.py）
# =========================

_number_token_pat = re.compile(r"-?\d")  # 至少包含一位数字（支持负号）
_number_clean_pat = re.compile(r"[^\d,\.-]")  # 清理掉除数字、逗号、小数点、负号以外的符号


def extract_logiqa_option(completion: str) -> str:
    """从模型输出中抽取 LogiQA 的多选项答案（A/B/C/D）。

    兼容形式：
      "Answer: D.", "The correct answer is: C.", "D. xxxx Explanation", "The correct answer is D"
    """
    if re.match(r"^[A-D]\.\s", completion):
        return completion[0]
    if all(k in completion for k in ["Answer:", "Explanation"]):
        m = re.search(r"Answer:\s*(.*?)\s*Explanation", completion, re.DOTALL)
        tmp = m.group(1) if m else re.split(r"Answer:\s*", completion, maxsplit=1)[1]
        m2 = re.search(r"\b[A-D]\b", tmp)
        return m2.group() if m2 else ""
    if "Explanation" in completion:
        tmp = re.split(r"Explanation", completion)[0]
        m = re.search(r"\b[A-D]\b", tmp)
        return m.group() if m else ""
    if re.search(r"(Answer:|answer is)\s*", completion):
        tmp = re.split(r"(?:Answer:|answer is)\s*", completion, maxsplit=1)[1]
        m = re.search(r"\b[A-D]\b", tmp)
        return m.group() if m else ""
    m = re.search(r"\b[A-D]\b", completion)
    return m.group() if m else ""


def extract_gsm8k_answer_number(completion: str) -> Optional[str]:
    """
    从任意输出中抽取 GSM8K 的最终数值答案。
    策略：分词 -> 仅保留含数字的 token -> 清理符号 -> 取最后一个 -> 规整成整数串。
    """
    tokens = re.split(r"[\s\n]+", completion)
    tokens_with_numbers = [t for t in tokens if _number_token_pat.search(t)]
    cleaned = [_number_clean_pat.sub("", t) for t in tokens_with_numbers]
    if not cleaned:
        return None
    x = cleaned[-1].replace(",", "").strip(".")
    if x.count(".") > 1 or x.count("-") > 1 or not re.match(r"^-?\d*\.?\d*$", x):
        return None
    try:
        return str(round(float(x)))
    except Exception:
        return None


def _post_process_latex_scalar(s: str) -> str:
    """
    对 LaTeX 形式的标量进行轻量后处理，使其与标注答案的归一化口径一致。
    包括：去角度标记、去美元符、去百分号、去空格。
    """
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")  # 去角度
    s = s.replace("\\$", "")  # 去美元符
    s = s.replace("\\%", "").replace("\%", "")  # 去百分号
    s = s.replace(" ", "")  # 去空格
    return s


def extract_math_answer(completion: str, label: str) -> Optional[str]:
    """
    从 MetaMathQA 输出中抽取最终答案。
    规则优先级：
      1) 若 gold label 含 \\text{...} 或含冒号/括号等自然语言成分，则做“反向匹配”：
         在输出末段（Therefore/So 之后）直接查找 label 的关键内容，命中则返回标准答案。
      2) 否则优先抽取最后一个 \boxed{...}。
      3) 若没有 \boxed，尝试抽取最后一个 \frac{d}{d} 形式。
      4) 仍失败则回退为 GSM8K 的数值抽取。
    """
    if any(k in label for k in ["\\text{", ":"]) or label.startswith("("):
        content = None
        if "\\text{" in label:
            m = re.search(r"\\text{(.*?)}", label)
            content = m.group(1) if m else None
        elif ":" in label or label.startswith("("):
            content = re.sub(r"^\(|\)$", "", label)
        parts = re.split(r"[Tt]herefore|[Ss]o", completion)
        if len(parts) > 1 and content and content in parts[-1].replace(" ", ""):
            return label
        return None

    m = re.search(r"\\boxed\{(.*)\}", completion)  # 贪婪匹配，确保捕获嵌套内的完整表达式
    if m:
        return _post_process_latex_scalar(m.group(1))

    fracs = re.findall(r"\\frac\{\d\}\{\d\}", completion)
    if fracs:
        return _post_process_latex_scalar(fracs[-1])

    return extract_gsm8k_answer_number(completion)


# =========================
# BERTScore / ROUGE / BLEU / BLEURT 计算（与官方脚本口径一致）
# =========================


def _ensure_nltk():
    """
    确保 NLTK 的分句与分词资源可用；若缺失则尝试静默下载。
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    # 新版 NLTK 可能需要 punkt_tab
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    return nltk


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    try:
        import torch
        import bert_score
        from bert_score.utils import get_tokenizer
    except ImportError:
        raise ImportError("请安装 bert_score 与 torch：pip install bert-score torch")

    max_tokens = 500
    tok = get_tokenizer(model_type="bert-base-multilingual-cased")

    cand = list(predictions)
    ref = list(references)

    for i in tqdm(range(len(cand)), desc="Pre-tokenizing for BERTScore", leave=False):
        c = cand[i]
        r = ref[i]
        ctoks = tok.tokenize(tok.decode(tok.encode(c, add_special_tokens=True)))
        rtoks = tok.tokenize(tok.decode(tok.encode(r, add_special_tokens=True)))
        if len(ctoks) > max_tokens or len(rtoks) > max_tokens:
            ctoks = ctoks[:max_tokens]
            rtoks = rtoks[:max_tokens]
            c_str = tok.convert_tokens_to_string(ctoks)
            r_str = tok.convert_tokens_to_string(rtoks)
            mlen = min(len(c_str), len(r_str))
            cand[i] = c_str[:mlen]
            ref[i] = r_str[:mlen]

    # 关键修复：不要再查 globals()，直接看 torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = bert_score.score(
        cand, ref, lang="en", model_type="bert-base-multilingual-cased", device=device, batch_size=16, verbose=True
    )
    return float(F1.mean().item())


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """计算 ROUGE（rouge1/rouge2/rougeL/rougeLsum 的 F1）并对样本平均，返回 0–1 的均值字典。.

    与官方一致的要点：
      - 候选摘要先做 pre_rouge_processing：把 <n> 替换为空格，然后句子切分并用换行连接；
      - 使用 rouge_score.RougeScorer，use_stemmer=True，split_summaries=True。
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("请安装 rouge_score：pip install rouge-score")

    nltk = _ensure_nltk()
    from nltk import sent_tokenize

    def _preprocess(summary: str) -> str:
        summary = summary.replace("<n>", " ")
        # 标准路径：用 NLTK 做句子切分，并用换行连接，符合 rougeLsum 的句级 LCS 口径
        try:
            return "\n".join(sent_tokenize(summary))
        except Exception:
            # 退化路径：用标点粗略切分
            pieces = re.split(r"[。.!?]+", summary)
            return "\n".join(x.strip() for x in pieces if x.strip())

    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True, split_summaries=True)

    r1 = r2 = rL = rLsum = 0.0
    for i in tqdm(range(len(predictions)), desc="Evaluating ROUGE", leave=False):
        scores = scorer.score(references[i], _preprocess(predictions[i]))
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rL += scores["rougeL"].fmeasure
        rLsum += scores["rougeLsum"].fmeasure

    n = max(len(predictions), 1)
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rL / n, "rougeLsum": rLsum / n}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """计算 NLTK sentence_bleu 的平均分（带 method4 平滑），返回 0–1 之间的均值。."""
    nltk = _ensure_nltk()
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    bleu_sum = 0.0
    for i in tqdm(range(len(predictions)), desc="Evaluating BLEU", leave=False):
        try:
            ref_tok = [nltk.tokenize.word_tokenize(references[i])]
            can_tok = nltk.tokenize.word_tokenize(predictions[i])
        except Exception:
            # 若分词器不可用，退化为空格切分
            ref_tok = [references[i].split()]
            can_tok = predictions[i].split()
        smooth = SmoothingFunction().method4
        bleu_sum += sentence_bleu(ref_tok, can_tok, smoothing_function=smooth)

    return bleu_sum / max(len(predictions), 1)


def _configure_tf_memory_growth():
    """配置 TensorFlow 的 GPU 显存按需增长。若未安装 TensorFlow，直接返回。."""
    try:
        import tensorflow as tf
    except ImportError:
        return
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except Exception as e:
        print(f"TensorFlow 显存增长设置失败：{e}")


def compute_bleurt(predictions: list[str], references: list[str]) -> float:
    """
    计算 BLEURT 的平均分，返回 0–1 之间的均值。
    模型：bleurt-base-128。优先从本地缓存与环境变量所指目录加载 checkpoint。
    """
    try:
        from bleurt import score as bleurt_score
    except ImportError:
        raise ImportError("请安装 BLEURT：pip install bleurt==0.0.2 tensorflow")

    _configure_tf_memory_growth()

    model_name = "bleurt-base-128"
    search_dirs = []
    env_cache = os.getenv("BLEURT_CACHE")
    if env_cache:
        search_dirs.append(os.path.expanduser(os.path.join(env_cache, model_name)))
    search_dirs.append(os.path.expanduser(os.path.join("~", ".bleurt", model_name)))
    search_dirs.append(os.path.join(os.getcwd(), model_name))

    marker_files = ("bleurt_config.json", "saved_model.pb")
    ckpt_path = None
    for d in search_dirs:
        if os.path.isdir(d) and all(os.path.exists(os.path.join(d, f)) for f in marker_files):
            ckpt_path = d
            break

    scorer = bleurt_score.BleurtScorer(ckpt_path) if ckpt_path else bleurt_score.BleurtScorer(model_name)

    scores = []
    for i in tqdm(range(len(predictions)), desc="Evaluating BLEURT", leave=False):
        s = scorer.score(references=[references[i]], candidates=[predictions[i]])
        scores.append(float(s[0]))

    return float(mean(scores)) if scores else 0.0


# =========================
# Accuracy（官方 eval_accuracy 口径）
# =========================


def calculate_accuracy(predictions: list[str], references: list[str], dataset_type: str) -> float:
    """
    计算分类准确率（Exact Match），返回 0–1 之间的均值。
    - logiqa：抽取 A/B/C/D；
    - gsm8k：抽取末尾数值；
    - metamathqa：\boxed 优先，其次 \frac，再次数值回退；含文本标签的反向匹配。
    - 其它：退化为数值相等或小写字符串相等的比较（仅兜底）。
    """
    correct = 0
    total = len(predictions)

    for pred_str, label_str in tqdm(
        zip(predictions, references), desc=f"Calculating Accuracy for '{dataset_type}'", total=total
    ):
        if dataset_type == "logiqa":
            pred = extract_logiqa_option(pred_str)
            gold = label_str.strip()
            correct += int(pred == gold)

        elif dataset_type == "gsm8k":
            pred = extract_gsm8k_answer_number(pred_str)
            gold = label_str.strip()
            correct += int(pred is not None and pred == gold)

        elif dataset_type == "metamathqa":
            pred = extract_math_answer(pred_str, label_str)
            gold = label_str.strip()
            correct += int(pred is not None and pred == gold)

        else:
            # 默认兜底：尽量不引入强假阳性
            try:
                p = float(pred_str.strip())
                g = float(label_str.strip())
                correct += int(p == g)
            except Exception:
                correct += int(pred_str.strip().lower() == label_str.strip().lower())

    return correct / total if total else 0.0


# =========================
# 主流程
# =========================


def main(filename: str, output_filename: str, *, metrics="all"):
    """
    加载数据、计算并保存评估结果（JSON 键为你原始格式，**数值以百分制写入与打印**）。

    参数：
        filename (str): 输入 JSON/JSONL 路径，样本需含字段 "predict" 与 "label"。
        output_filename (str): 输出评估结果的 JSON 路径。
        metrics (str | list | tuple): 选择计算的指标。逗号分隔字符串或多参数皆可。
            支持： "bertscore"|"bs", "rouge", "bleu", "em"|"exact_match", "bleurt"
            特别地："all" 只计算 bertscore / rouge / bleu / em（与官方默认一致），不包含 bleurt。

    说明：
        - 计算阶段内部均按 0–1 范围进行；在**最终写文件与打印时**统一乘以 100，并四舍五入到 4 位小数。
        - JSON 键名保持为：
            "bertscore-f1", "rouge-1", "rouge-2", "rouge-l", "rouge-Lsum", "bleu", "exact_match"
          若选择 bleurt，则额外包含键 "bleurt"。
    """
    start_time = time.time()

    # --- 解析指标参数 ---
    if isinstance(metrics, str):
        metric_list = [m.strip().lower() for m in metrics.split(",")]
    elif isinstance(metrics, (list, tuple)):
        metric_list = [str(m).strip().lower() for m in metrics]
    else:
        metric_list = []

    metric_map = {
        "bs": "bertscore",
        "bertscore": "bertscore",
        "rouge": "rouge",
        "bleu": "bleu",
        "em": "em",
        "exact_match": "em",
        "bleurt": "bleurt",
    }
    if "all" in metric_list or not metric_list:
        enabled = {"bertscore", "rouge", "bleu", "em"}  # 与官方一致；不含 bleurt
    else:
        enabled = {metric_map[m] for m in metric_list if m in metric_map}

    print(f"将要计算的指标: {', '.join(sorted(list(enabled)))}")

    # --- 加载数据 ---
    print("\n正在加载数据集...")
    try:
        dataset = load_dataset("json", data_files=filename, split="train")
    except Exception as e:
        print(f"无法加载文件 {filename}。请检查文件路径和格式。错误: {e}")
        return

    predictions = [sample["predict"] for sample in dataset]
    references = [sample["label"] for sample in dataset]

    # --- 推断数据集类型（用于 Accuracy 的抽取逻辑） ---
    fn_lower = filename.lower()
    dataset_type = "default"
    if "logiqa" in fn_lower:
        dataset_type = "logiqa"
    elif "gsm8k" in fn_lower:
        dataset_type = "gsm8k"
    elif "metamath" in fn_lower or "metamathqa" in fn_lower:
        dataset_type = "metamathqa"
    print(f"检测到数据集类型: {dataset_type}")

    # --- 结果键名保持为原始 JSON 格式；最终数值写入为百分制（乘以 100） ---
    results: dict = {}

    if "bertscore" in enabled:
        print("\n正在计算 BERTScore...")
        v = compute_bertscore(predictions, references)  # 0–1
        results["bertscore-f1"] = round(v * 100, 4)  # 百分制
        print("BERTScore 完成。")

    if "rouge" in enabled:
        print("\n正在计算 ROUGE...")
        r = compute_rouge(predictions, references)  # 0–1
        results["rouge-1"] = round(r["rouge1"] * 100, 4)
        results["rouge-2"] = round(r["rouge2"] * 100, 4)
        results["rouge-l"] = round(r["rougeL"] * 100, 4)
        results["rouge-Lsum"] = round(r["rougeLsum"] * 100, 4)
        print("ROUGE 完成。")

    if "bleu" in enabled:
        print("\n正在计算 BLEU...")
        v = compute_bleu(predictions, references)  # 0–1
        results["bleu"] = round(v * 100, 4)
        print("BLEU 完成。")

    if "em" in enabled:
        print("\n正在计算 Accuracy（Exact Match）...")
        v = calculate_accuracy(predictions, references, dataset_type)  # 0–1
        results["exact_match"] = round(v * 100, 4)
        print("Accuracy 完成。")

    if "bleurt" in enabled:
        print("\n正在计算 BLEURT...")
        v = compute_bleurt(predictions, references)  # 0–1
        results["bleurt"] = round(v * 100, 4)  # 可选指标
        print("BLEURT 完成。")

    # --- 打印与保存（百分制） ---
    print("\n--- 最终评估结果（百分制 0–100）---")
    if not results:
        print("没有计算任何指标。请检查 'metrics' 参数。")
    else:
        for k in sorted(results.keys()):
            print(f"{k:<15}: {results[k]:.4f}")
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n分数文件已保存至: {output_filename}")

    print(f"\n计算完成，总用时 {time.time() - start_time:.3f} 秒。")


if __name__ == "__main__":
    # 使用方式示例：
    #   python eval_aligned.py --filename path/to/preds.json --output_filename path/to/scores.json --metrics all
    #   python eval_aligned.py --filename path/to/preds.json --output_filename path/to/scores.json --metrics rouge,bs,em
    #   python eval_aligned.py --filename path/to/preds.json --output_filename path/to/scores.json --metrics bleurt
    fire.Fire(main)
