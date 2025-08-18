#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
from pathlib import Path
import sys
from io import StringIO
import re
import argparse


# =========================
#  基础：IO 与解析函数
# =========================

def _load_metrics_file(fp: Path) -> dict | None:
    """
    加载并解析单个JSON或JSONL指标文件。
    - 兼容大小写不一致的key，全部转为小写处理。
    - 优先尝试将整个文件作为单个JSON解析。
    - 如果失败，则尝试逐行解析（JSONL），只返回第一条有效记录。
    """
    try:
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            return None
        # ① 整体解析
        try:
            return {k.lower(): v for k, v in json.loads(text).items()}
        except json.JSONDecodeError:
            pass

        # ② JSONL：逐行解析，取第一条
        for line in StringIO(text):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                return {k.lower(): v for k, v in obj.items()}
            except json.JSONDecodeError:
                sys.stderr.write(f"[警告] 跳过无法解析的行: {line} in file {fp}\n")
                continue

        return None
    except Exception as e:
        sys.stderr.write(f"[跳过] 处理 '{fp}' 出错：{e}\n")
        return None


def _load_llm_em_from_csv(fp: Path) -> float | None:
    """
    从 *_LLM_EM_summary.csv 中加载 LLM_EM 分数。
    - 查找包含 'match_rate' 且包含 'is_true_rate' 的列。
    - 支持百分号字符串与数值。
    """
    try:
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()
        target_col = next((c for c in df.columns if "match_rate" in c and "is_true_rate" in c), None)
        if target_col is None:
            sys.stderr.write(f"[警告] 在 {fp} 中未找到 'match_rate (is_true_rate)' 列。\n")
            return None
        val = df[target_col].iloc[0]
        if isinstance(val, str) and val.endswith("%"):
            return round(float(val.strip("%")), 2)
        return float(val)
    except Exception as e:
        sys.stderr.write(f"[跳过] 处理CSV文件 '{fp}' 出错: {e}\n")
        return None


# =========================
#  规则：benchmark / dataset “写死”匹配（不附加数据量）
# =========================

# 数据集同义词（小写）
_DATASET_SYNONYMS: dict[str, list[str]] = {
    # db 组
    "geosignal": ["geosignal"],
    "gen_med_gpt": ["gen_med_gpt", "genmedgpt", "gen-med-gpt", "genmed_gpt"],
    "agriculture": ["agriculture", "agriculture_qa", "agriculture-qa"],
    "wealth": ["wealth", "wealth_alpaca", "wealth-alpaca", "wealth_alpaca_lora", "wealth-alpaca-lora"],

    # ib 组
    "alpaca_gpt4": ["alpaca_gpt4", "alpaca_gpt4_en"],
    "instruction_wild": ["instruction_wild", "instructionwild"],
    "dolly": ["dolly", "dolly_15k", "dolly-15k"],

    # rb 组
    "gsm8k": ["gsm8k"],
    "logiqa": ["logiqa"],
    "meta_math": ["meta_math", "metamath", "metamathqa", "meta-math", "metamath_qa"],
}

# 基准分组写死
_DATASET_TO_BENCHMARK: dict[str, str] = {
    "geosignal": "db",
    "gen_med_gpt": "db",
    "agriculture": "db",
    "wealth": "db",
    "alpaca_gpt4": "ib",
    "instruction_wild": "ib",
    "dolly": "ib",
    "gsm8k": "rb",
    "logiqa": "rb",
    "meta_math": "rb",
}

# 识别测试/验证集（仅用于决定是否考虑规模，但现在我们不再附加规模，保留此标志以便未来扩展）
_EVAL_SPLIT_TOKENS = re.compile(r'(?:^|[/_\-])(test|valid|validation)(?:$|[/_\-])', re.IGNORECASE)

# 仅用于清理：去掉诸如末尾的 _5k / -5k（如果有的话）
_TRAILING_SIZE = re.compile(r'([_\-])\d+\s*[kK]$')


def _normalize_for_match(s: str) -> str:
    """
    统一小写；把非字母数字的分隔符统一成下划线；保留路径分隔 / 以便识别所在目录。
    """
    s = s.lower().replace("\\", "/")
    s = re.sub(r'[^\w/]+', '_', s)          # 其他符号 → _
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def _collect_candidate_strings(fp: Path) -> list[str]:
    """
    为匹配收集候选字符串（标准化）：
    - 全路径
    - 文件名
    - 文件干名（去除扩展）及去掉常见后缀的干名
    - 两级父目录名
    """
    full = _normalize_for_match(str(fp))
    name = _normalize_for_match(fp.name)
    stem = _normalize_for_match(fp.stem)
    stem_clean = re.sub(r'_(metrics|llm_em_summary)$', '', stem)

    parents = [p.name for p in fp.parents if p != fp.anchor]
    parents = list(reversed(parents))
    parent_norms = [_normalize_for_match(p) for p in parents[:3]]

    cands = [full, name, stem, stem_clean] + parent_norms
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _strip_trailing_size(token: str) -> str:
    """去掉末尾的 _5k / -5k 等数量标记。"""
    return _TRAILING_SIZE.sub('', token)


def _find_dataset_base(fp: Path) -> str:
    """
    从路径与名称候选集中，找到数据集“基础名”（不带数据量后缀）。
    """
    candidates = _collect_candidate_strings(fp)

    # 先尝试同义词
    for base, synonyms in _DATASET_SYNONYMS.items():
        for syn in synonyms:
            syn_norm = syn.lower().replace("-", "_")
            if any(syn_norm in c for c in candidates):
                return base

    # 尝试旧式命名：db_xxx / ib_xxx / rb_xxx
    for c in candidates:
        m = re.search(r'(^|[/_\-])(db|ib|rb)[_\-]([a-z0-9_]+)', c)
        if m:
            raw = m.group(3).strip('_-')
            raw = _strip_trailing_size(raw)
            # 再映射回标准 base
            for base, synonyms in _DATASET_SYNONYMS.items():
                if any(s in raw for s in synonyms):
                    return base
            return raw  # 未识别则直接返回 raw（无数量后缀）

    # 兜底：从文件干名推断
    stem = _normalize_for_match(fp.stem)
    stem = re.sub(r'_(metrics|llm_em_summary)$', '', stem)
    stem = _strip_trailing_size(stem)
    # 同义词再试一次
    for base, synonyms in _DATASET_SYNONYMS.items():
        if any(s in stem for s in synonyms):
            return base
    # 否则取第一段
    parts = [p for p in re.split(r'[_\-]+', stem) if p]
    return parts[0] if parts else "unknown"


def infer_benchmark_and_dataset(fp: Path) -> tuple[str, str]:
    """
    根据文件路径/名称推断 (benchmark, dataset)。
    - benchmark 由数据集基础名经写死表得到（db/ib/rb）；若不在表中则为 'unknown'。
    - dataset 为基础名本身（不带数据量）。
    """
    base = _find_dataset_base(fp)
    bench = _DATASET_TO_BENCHMARK.get(base, "unknown")
    dataset = base
    return bench, dataset


def infer_setting_from_parts(root_path: Path, parts: tuple[str, ...]) -> str:
    """
    setting 推断：
      - 若第二段存在且是目录：使用 "parts[0]/parts[1]"
      - 若第二段不存在或不是目录（多半是文件）：使用 "parts[0]"
      - 若没有任何段：返回 "."
    """
    if len(parts) >= 2:
        second_path = root_path / parts[0] / parts[1]
        if second_path.is_dir():
            return f"{parts[0]}/{parts[1]}"
        return parts[0]
    if len(parts) == 1:
        return parts[0]
    return "."


# =========================
#  主逻辑
# =========================

def analyze_metrics(root_dir: str) -> pd.DataFrame | None:
    """
    扫描指定根目录，解析所有指标文件（JSON/JSONL/CSV），并将结果整合到一个DataFrame中。
    - benchmark 与 dataset 来自“写死”的匹配规则（不拼接数量）。
    - setting 采用“智能两级”：第二段是目录时用两级，否则只用第一级。
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        sys.stderr.write(f"错误：目录 '{root_dir}' 不存在。\n")
        return None

    records_dict: dict[tuple[str, str, str], dict] = {}

    # -------- 解析 *_metrics.json / *_metrics.jsonl --------
    for fp in root_path.rglob("*_metrics.json*"):
        parts = fp.relative_to(root_path).parts
        setting = infer_setting_from_parts(root_path, parts)
        benchmark, dataset = infer_benchmark_and_dataset(fp)

        metrics = _load_metrics_file(fp)
        if metrics is None:
            continue

        rec_key = (benchmark, dataset, setting)
        if rec_key not in records_dict:
            records_dict[rec_key] = {"benchmark": benchmark, "dataset": dataset, "setting": setting}
        for k, v in metrics.items():
            records_dict[rec_key][k] = v

    # -------- 解析 *_LLM_EM_summary.csv（可选）--------
    for fp in root_path.rglob("*LLM_EM_summary.csv"):
        parts = fp.relative_to(root_path).parts
        setting = infer_setting_from_parts(root_path, parts)
        benchmark, dataset = infer_benchmark_and_dataset(fp)
        llm_em_score = _load_llm_em_from_csv(fp)
        if llm_em_score is None:
            continue

        rec_key = (benchmark, dataset, setting)
        if rec_key not in records_dict:
            records_dict[rec_key] = {"benchmark": benchmark, "dataset": dataset, "setting": setting}
            sys.stderr.write(f"[信息] 为 {fp} 创建了新记录（仅含 LLM_EM）。\n")
        records_dict[rec_key]["llm_em"] = llm_em_score

    if not records_dict:
        print("未找到任何可用 metrics 文件。")
        return None

    # -------- 构建 DataFrame --------
    records = list(records_dict.values())
    df = pd.DataFrame(records).set_index(["benchmark", "dataset", "setting"]).sort_index()

    # 尝试将所有列转成数值，无法转换的保留为 NaN（便于聚合/比较）
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 保存汇总结果
    df.to_csv("result.csv")
    return df


def display_filtered_results(
    df: pd.DataFrame,
    setting_pattern: str | None = None,
    benchmark_filter: list[str] | None = None,
    dataset_filter: list[str] | None = None,
):
    """
    根据指定的筛选条件，在控制台打印格式化的DataFrame。
    - 筛选结果会自动包含 'base' 方法便于对比（若存在）。
    - setting 支持正则匹配；benchmark/dataset 支持列表过滤。
    """
    if df is None or df.empty:
        print("输入的DataFrame为空，无法进行筛选和显示。")
        return

    if not any([setting_pattern, benchmark_filter, dataset_filter]):
        print("\n--- 完整指标汇总 (无筛选) ---")
        with pd.option_context(
            "display.float_format", "{:.4f}".format,
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 200,
        ):
            print(df)
        return

    temp_df = df.reset_index()
    filtered_df = temp_df.copy()

    if setting_pattern:
        try:
            is_base = filtered_df["setting"] == "base"
            matches = filtered_df["setting"].str.match(setting_pattern, na=False)
            filtered_df = filtered_df[is_base | matches]
        except re.error as e:
            print(f"[错误] setting筛选的正则表达式无效: {e}")
            return

    if benchmark_filter:
        filtered_df = filtered_df[filtered_df["benchmark"].isin(benchmark_filter)]

    if dataset_filter:
        filtered_df = filtered_df[filtered_df["dataset"].isin(dataset_filter)]

    if filtered_df.empty:
        print("\n根据您的筛选条件，没有找到任何匹配的数据。")
        return

    related_benchmarks = filtered_df["benchmark"].unique()
    related_datasets = filtered_df["dataset"].unique()

    base_to_display = temp_df[
        (temp_df["setting"] == "base")
        & (temp_df["benchmark"].isin(related_benchmarks))
        & (temp_df["dataset"].isin(related_datasets))
    ]

    final_df_rows = pd.concat([base_to_display, filtered_df]).drop_duplicates()
    display_df = final_df_rows.set_index(["benchmark", "dataset", "setting"]).sort_index()

    print("\n--- 筛选结果 (包含 'base' 对比) ---")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
    ):
        print(display_df)


# =========================
#  CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="分析模型性能指标并根据条件筛选结果。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-r", "--root-dir", type=str,
        help="指定要分析的根目录。\n如果未提供，将默认使用脚本所在目录的父目录下的 'results' 文件夹。",
    )
    parser.add_argument(
        "-s", "--setting-pattern", type=str,
        help="用于筛选 'setting' 的正则表达式。\n示例: 'ttl.*' 会匹配以 ttl 开头的设置。",
    )
    parser.add_argument(
        "-b", "--benchmark-filter", nargs="+", type=str,
        help="要显示的 benchmark 列表（如：-b db ib rb）。",
    )
    parser.add_argument(
        "-d", "--dataset-filter", nargs="+", type=str,
        help="要显示的 dataset 列表（如：-d gsm8k alpaca_gpt4）。",
    )

    args = parser.parse_args()

    # 根目录推断
    if args.root_dir:
        root_dir = Path(args.root_dir).resolve()
    else:
        try:
            root_dir = Path(__file__).resolve().parent.parent / "results"
        except NameError:
            root_dir = Path("./results").resolve()

    print(f"--- 开始分析目录: {root_dir} ---")
    full_df = analyze_metrics(str(root_dir))

    if full_df is not None:
        display_filtered_results(
            full_df,
            setting_pattern=args.setting_pattern,
            benchmark_filter=args.benchmark_filter,
            dataset_filter=args.dataset_filter,
        )
    else:
        print("未能生成DataFrame，无法显示结果。")


if __name__ == "__main__":
    main()