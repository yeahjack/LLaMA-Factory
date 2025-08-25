#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA safetensors 目录分析器
- 输入一个目录，自动定位 adapter_model.safetensors
- 计算各模块有效更新 ||ΔW||_F，并做多种维度的聚合
用法示例：
  python analyze_lora_dir.py /home/yijiexu/LLaMA-Factory/saves/qwen25_7b/ttl-offline_ttl_thr3_sc0p1/agriculture_5k
常用可选参数：
  --which latest|top|all|step   默认 latest（优先最高 step 的 checkpoint；若无 checkpoint，则用顶层）
  --step 5000                   当 --which step 时指定
  --topk 30                     打印模块级前 topk
  --by module|layer|block|proj  选择额外的聚合视图，默认都会打印
"""

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import torch
from safetensors.torch import load_file

# --------------------------- 路径与元数据 ---------------------------

_ckpt_pat = re.compile(r"checkpoint-(\d+)")

@dataclass
class AdapterFile:
    path: str
    step: Optional[int]  # 从目录名提取的 step；若无则为 None
    where: str           # 'checkpoint' 或 'top'

def find_adapter_files(root: str) -> list[AdapterFile]:
    files: list[AdapterFile] = []
    for dirpath, _, filenames in os.walk(root):
        if "adapter_model.safetensors" in filenames:
            fpath = os.path.join(dirpath, "adapter_model.safetensors")
            m = _ckpt_pat.search(dirpath)
            step = int(m.group(1)) if m else None
            where = "checkpoint" if m else ("top" if os.path.abspath(dirpath) == os.path.abspath(root) else "other")
            files.append(AdapterFile(fpath, step, where))
    return files

def choose_files(files: list[AdapterFile], which: str, step: Optional[int]) -> list[AdapterFile]:
    if not files:
        return []
    if which == "all":
        return sorted(files, key=lambda x: (x.step is None, -(x.step or -1), x.path))
    if which == "step":
        if step is None:
            raise ValueError("使用 --which step 需要提供 --step")
        # 优先匹配 checkpoint-<step>，否则报错
        matches = [f for f in files if f.step == step]
        if not matches:
            raise FileNotFoundError(f"未找到 checkpoint-{step} 的 adapter_model.safetensors")
        return sorted(matches, key=lambda x: x.path)
    if which == "top":
        tops = [f for f in files if f.where == "top"]
        if tops:
            return sorted(tops, key=lambda x: x.path)
        # 没有顶层就退而选 latest
        which = "latest"
    if which == "latest":
        with_step = [f for f in files if f.step is not None]
        if with_step:
            best = max(with_step, key=lambda x: x.step)  # 最高步数
            return [best]
        # 没有 checkpoint 的情况，选顶层优先，否则选任意一个
        tops = [f for f in files if f.where == "top"]
        return [tops[0]] if tops else [sorted(files, key=lambda x: x.path)[0]]
    raise ValueError(f"未知 --which 选项: {which}")

# --------------------------- 权重解析与范数 ---------------------------

# 兼容常见命名：*.lora_A.weight, *.lora_B.weight, *.lora_alpha 或 *.alpha
_re_A = re.compile(r"^(.*?)(?:\.)?lora_A\.weight$")
_re_B = re.compile(r"^(.*?)(?:\.)?lora_B\.weight$")
_re_alpha = re.compile(r"^(.*?)(?:\.)?(?:lora_)?alpha$")

@dataclass
class LoRAGroup:
    A: Optional[torch.Tensor] = None   # 形状期望 (r, in)
    B: Optional[torch.Tensor] = None   # 形状期望 (out, r)
    alpha: Optional[float] = None

def load_lora_groups(path: str) -> dict[str, LoRAGroup]:
    tensors = load_file(path)
    groups: dict[str, LoRAGroup] = defaultdict(LoRAGroup)
    for k, v in tensors.items():
        m = _re_A.match(k)
        if m:
            groups[m.group(1)].A = v
            continue
        m = _re_B.match(k)
        if m:
            groups[m.group(1)].B = v
            continue
        m = _re_alpha.match(k)
        if m:
            try:
                groups[m.group(1)].alpha = float(v.item())
            except Exception:
                # alpha 可能是张量而非标量，这里取均值以保证健壮性
                groups[m.group(1)].alpha = float(v.float().mean().item())
            continue
    return groups

def align_AB(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 期望 A:(r,in), B:(out,r)。若不匹配则尝试转置 A。
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A/B 张量维度异常")
    if B.shape[1] == A.shape[0]:
        return A, B
    if B.shape[1] == A.shape[1]:
        return A.t(), B
    # 有些实现把 A/B 方向都对调了（少见），再尝试一次
    if B.shape[0] == A.shape[0]:
        return A, B.t()
    if B.shape[0] == A.shape[1]:
        return A.t(), B.t()
    raise ValueError(f"A/B 形状无法对齐: A{tuple(A.shape)} B{tuple(B.shape)}")

@dataclass
class ModuleDelta:
    name: str
    fro_norm: float
    rank: int

def delta_fro_norm(A: torch.Tensor, B: torch.Tensor, scale: float) -> float:
    # 避免显式构造 BA：||BA||_F^2 = tr(B^T B · A A^T)
    A = A.to(dtype=torch.float32, copy=False)
    B = B.to(dtype=torch.float32, copy=False)
    BtB = B.transpose(0, 1) @ B          # (r, r)
    AAt = A @ A.transpose(0, 1)          # (r, r)
    fro2 = (scale ** 2) * torch.trace(BtB @ AAt).item()
    return float(fro2 ** 0.5)

def analyze_adapter_file(path: str) -> list[ModuleDelta]:
    groups = load_lora_groups(path)
    results: list[ModuleDelta] = []
    for base, g in groups.items():
        if g.A is None or g.B is None:
            continue
        A, B = align_AB(g.A, g.B)
        r = A.shape[0]
        alpha = g.alpha if g.alpha is not None else r  # 与 PEFT 一致：scale = alpha / r；若无 alpha 视作 alpha=r → scale=1
        scale = float(alpha) / float(r)
        fro = delta_fro_norm(A, B, scale)
        results.append(ModuleDelta(base, fro, r))
    results.sort(key=lambda x: x.fro_norm, reverse=True)
    return results

# --------------------------- 解析模块归类 ---------------------------

_layer_pat = re.compile(r"\.layers\.(\d+)\.")
def get_layer_idx(name: str) -> Optional[int]:
    m = _layer_pat.search(name)
    return int(m.group(1)) if m else None

def get_block_type(name: str) -> str:
    s = name.lower()
    if "attn" in s or "attention" in s:
        return "attn"
    if "mlp" in s or "ffn" in s or "feed_forward" in s:
        return "ffn"
    return "other"

def get_proj_type(name: str) -> str:
    s = name.lower()
    for p in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]:
        if p in s:
            return p
    return "other"

@dataclass
class Aggregates:
    by_layer: dict[int, float]
    by_block: dict[str, float]
    by_proj: dict[str, float]
    total: float

def aggregate(results: list[ModuleDelta]) -> Aggregates:
    by_layer: dict[int, float] = defaultdict(float)
    by_block: dict[str, float] = defaultdict(float)
    by_proj: dict[str, float] = defaultdict(float)
    total = 0.0
    for r in results:
        w = r.fro_norm
        total += w
        li = get_layer_idx(r.name)
        if li is not None:
            by_layer[li] += w
        by_block[get_block_type(r.name)] += w
        by_proj[get_proj_type(r.name)] += w
    return Aggregates(by_layer, by_block, by_proj, total)

def pct(x: float, total: float) -> float:
    return (100.0 * x / total) if total > 0 else 0.0

# --------------------------- CLI 与打印 ---------------------------

def print_summary(results: list[ModuleDelta], topk: int) -> None:
    print("\n[模块级 Top-{}  (按 ||ΔW||_F 排序)]".format(topk))
    for i, r in enumerate(results[:topk], 1):
        print(f"{i:>3}. {r.name}   rank={r.rank:<3}  ||ΔW||_F={r.fro_norm:.6f}")

    aggr = aggregate(results)
    print("\n[按块类型汇总（attn / ffn / other）]")
    for k, v in sorted(aggr.by_block.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:<6}  sum={v:.6f}  share={pct(v, aggr.total):5.2f}%")

    print("\n[按投影汇总（q/k/v/o 与 up/gate/down）]")
    for k, v in sorted(aggr.by_proj.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:<9}  sum={v:.6f}  share={pct(v, aggr.total):5.2f}%")

    if aggr.by_layer:
        print("\n[按层号汇总（sum ||ΔW||_F）]")
        for li, v in sorted(aggr.by_layer.items(), key=lambda kv: kv[0]):
            print(f"layer {li:>2}:  {v:.6f}  share={pct(v, aggr.total):5.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="包含 LoRA adapter 的目录")
    parser.add_argument("--which", type=str, default="latest",
                        choices=["latest", "top", "all", "step"],
                        help="选择哪个 adapter 分析")
    parser.add_argument("--step", type=int, default=None, help="当 --which step 时指定的步数")
    parser.add_argument("--topk", type=int, default=30, help="打印模块级前 N")
    args = parser.parse_args()

    files = find_adapter_files(args.dir)
    if not files:
        raise FileNotFoundError("目录内未找到任何 adapter_model.safetensors")

    chosen = choose_files(files, args.which, args.step)
    print(f"找到 {len(files)} 个 adapter；本次分析 {len(chosen)} 个：")
    for f in chosen:
        tag = f"checkpoint-{f.step}" if f.step is not None else f.where
        print(f" - [{tag}] {f.path}")

    for idx, f in enumerate(chosen, 1):
        print("\n" + "=" * 80)
        title = f"checkpoint-{f.step}" if f.step is not None else f.where
        print(f"[{idx}/{len(chosen)}] 分析：{title}")
        results = analyze_adapter_file(f.path)
        if not results:
            print("未解析到任何 LoRA A/B 对；可能是已合并权重或非常规命名。")
            continue
        print_summary(results, args.topk)

if __name__ == "__main__":
    main()