import pandas as pd
import json
from pathlib import Path
import sys

def _load_metrics_file(fp: Path) -> dict | None:
    """智能加载 .json 或 .jsonl 指标文件。"""
    try:
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            return None
        # ① 尝试整体解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # ② 逐行解析（真正的 JSONL）
        objs = []
        with fp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    objs.append(json.loads(line))
        if not objs:
            return None
        if len(objs) > 1:
            sys.stderr.write(f"[警告] {fp} 含多条 JSONL，只取第一条\n")
        return objs[0]
    except Exception as e:
        sys.stderr.write(f"[跳过] 处理 '{fp}' 出错：{e}\n")
        return None


def analyze_metrics(root_dir: str = "results_no_system"):
    root_path = Path(root_dir)
    if not root_path.is_dir():
        sys.stderr.write(f"错误：目录 '{root_dir}' 不存在。\n")
        return None

    records = []
    for fp in root_path.rglob("*_metrics.json*"):
        parts = fp.relative_to(root_path).parts
        # 两层或三层目录解析
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

        rec = {"setting": setting, "benchmark": benchmark, "dataset": dataset}
        rec.update(metrics)
        records.append(rec)

    if not records:
        print("未找到任何可用 metrics 文件。")
        return None

    df = (pd.DataFrame(records)
            .set_index(["setting", "benchmark", "dataset"])
            .sort_index())
    pd.set_option("display.float_format", "{:.4f}".format)
    # 显示所有行和列（不省略）
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    pd.set_option("display.max_colwidth", None)
    return df


if __name__ == "__main__":
    root_dir = Path(__file__).parent / "../results"
    print(root_dir)
    df = analyze_metrics(root_dir)
    if df is not None:
        print("\n--- 指标汇总 ---\n", df)