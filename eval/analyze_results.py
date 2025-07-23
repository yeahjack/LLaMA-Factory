import pandas as pd
import json
from pathlib import Path
import sys
from io import StringIO
import re
import argparse


def _load_metrics_file(fp: Path) -> dict | None:
    try:
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            return None
        # ① 尝试整体解析
        try:
            # 兼容大小写不一致的key，全部转为小写处理
            return {k.lower(): v for k, v in json.loads(text).items()}
        except json.JSONDecodeError:
            pass

        # ② 逐行解析（真正的 JSONL）
        objs = []
        # 使用 StringIO 来处理多行JSON文本
        for line in StringIO(text):
            line = line.strip()
            if line:
                try:
                    # 兼容大小写不一致的key，全部转为小写处理
                    objs.append({k.lower(): v for k, v in json.loads(line).items()})
                except json.JSONDecodeError:
                    sys.stderr.write(f"[警告] 跳过无法解析的行: {line} in file {fp}\n")
                    continue

        if not objs:
            return None
        if len(objs) > 1:
            sys.stderr.write(f"[警告] {fp} 含多条 JSONL，只取第一条\n")
        return objs[0]
    except Exception as e:
        sys.stderr.write(f"[跳过] 处理 '{fp}' 出错：{e}\n")
        return None


def _load_llm_em_from_csv(fp: Path) -> float | None:
    # ... (此函数无任何变化)
    try:
        df = pd.read_csv(fp)
        # 清理列名中的空格，以防万一
        df.columns = df.columns.str.strip()
        # 查找目标列，该列名包含 'match_rate' 和 'is_true_rate'
        target_col = next((col for col in df.columns if "match_rate" in col and "is_true_rate" in col), None)

        if target_col is None:
            sys.stderr.write(f"[警告] 在 {fp} 中未找到 'match_rate (is_true_rate)' 列。\n")
            return None

        # 提取值，例如 "95.91%"
        rate_str = df[target_col].iloc[0]

        if isinstance(rate_str, str) and rate_str.endswith("%"):
            # 移除 '%' 并转换为 0-1 之间的小数
            return round(float(rate_str.strip("%")), 2)
        else:
            # 如果值已经是小数形式，直接转换
            return float(rate_str)

    except Exception as e:
        sys.stderr.write(f"[跳过] 处理CSV文件 '{fp}' 出错: {e}\n")
        return None


def analyze_metrics(root_dir: str = "results_no_system") -> pd.DataFrame | None:
    """
    扫描目录，解析指标文件（JSON/JSONL/CSV），并返回一个DataFrame。
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        sys.stderr.write(f"错误：目录 '{root_dir}' 不存在。\n")
        return None

    records_dict = {}

    # --- 数据加载部分无变化 ---
    for fp in root_path.rglob("*_metrics.json*"):
        parts = fp.relative_to(root_path).parts
        if len(parts) == 3:
            setting, benchmark, fname = parts
        elif len(parts) == 2:
            setting, fname = parts
            benchmark = fname.split("_", 1)[0] if "_" in fname else "unknown"
        else:
            continue
        suffix = "_metrics.jsonl" if fname.endswith(".jsonl") else "_metrics.json"
        dataset = fname.removesuffix(suffix).split("_", 1)[-1]
        metrics = _load_metrics_file(fp)
        if metrics is None:
            continue
        rec_key = (setting, benchmark, dataset)
        if rec_key not in records_dict:
            records_dict[rec_key] = {"setting": setting, "benchmark": benchmark, "dataset": dataset}
        records_dict[rec_key].update(metrics)

    for fp in root_path.rglob("*_LLM_EM_summary.csv"):
        parts = fp.relative_to(root_path).parts
        if len(parts) == 3:
            setting, benchmark, fname = parts
        elif len(parts) == 2:
            setting, fname = parts
            benchmark = fname.split("_", 1)[0] if "_" in fname else "unknown"
        else:
            continue
        dataset = fname.removesuffix("_LLM_EM_summary.csv").split("_", 1)[-1]
        llm_em_score = _load_llm_em_from_csv(fp)
        if llm_em_score is None:
            continue
        rec_key = (setting, benchmark, dataset)
        if rec_key in records_dict:
            records_dict[rec_key]["llm_em"] = llm_em_score
        else:
            records_dict[rec_key] = {
                "setting": setting,
                "benchmark": benchmark,
                "dataset": dataset,
                "llm_em": llm_em_score,
            }
            sys.stderr.write(f"[信息] 为 {fp} 创建了新记录，因为它没有对应的 JSON 指标文件。\n")

    if not records_dict:
        print("未找到任何可用 metrics 文件。")
        return None

    records = list(records_dict.values())

    # 【关键修改点 1】: 更改 set_index 的列顺序以创建正确的 MultiIndex 层次结构
    df = pd.DataFrame(records).set_index(["benchmark", "dataset", "setting"])

    # 排序以确保分组正确
    df = df.sort_index()

    # 将所有非数值列转换为数值，无法转换的设为NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
    """
    if df is None or df.empty:
        print("输入的DataFrame为空，无法进行筛选和显示。")
        return

    # 如果没有任何筛选条件，则显示已按期望格式排序的完整报告
    if not any([setting_pattern, benchmark_filter, dataset_filter]):
        print("\n--- 完整指标汇总 (无筛选) ---")
        with pd.option_context(
            "display.float_format",
            "{:.4f}".format,
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            200,
        ):
            print(df)
        return

    temp_df = df.reset_index()

    filtered_df = temp_df.copy()
    if setting_pattern:
        try:
            # 确保不筛选掉 'base' 方法
            is_base = filtered_df["setting"] == "base"
            matches_pattern = filtered_df["setting"].str.match(setting_pattern, na=False)
            filtered_df = filtered_df[is_base | matches_pattern]
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

    # 【关键修改点 2】: 使用与默认显示完全相同的索引结构
    display_df = final_df_rows.set_index(["benchmark", "dataset", "setting"])

    # 排序以匹配默认的显示格式
    display_df = display_df.sort_index()

    print("\n--- 筛选结果 (包含 'base' 对比) ---")
    with pd.option_context(
        "display.float_format",
        "{:.4f}".format,
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(display_df)


if __name__ == "__main__":
    # ... (命令行解析部分无任何变化)
    parser = argparse.ArgumentParser(
        description="分析模型性能指标并根据条件筛选结果。", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--setting-pattern",
        type=str,
        help="用于筛选'方法'(setting)的正则表达式。\n示例: 'eata.*' 会匹配 'eata' 和 'eata_sdiv'。",
    )
    parser.add_argument(
        "-b",
        "--benchmark-filter",
        nargs="+",
        type=str,
        help="要显示的'基准'(benchmark)列表，用空格分隔。\n示例: -b db ib",
    )
    parser.add_argument(
        "-d",
        "--dataset-filter",
        nargs="+",
        type=str,
        help="要显示的'数据集'(dataset)列表，用空格分隔。\n示例: -d gsm8k alpaca_eval",
    )
    args = parser.parse_args()

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
