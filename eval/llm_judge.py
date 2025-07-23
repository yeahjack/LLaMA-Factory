import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

import httpx
import pandas as pd  # 为保存CSV而添加
import requests
import torch

# --- SGLang Imports (保留) ---
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server
from tqdm.asyncio import tqdm_asyncio


# ==============================================================================
# --- 核心配置 ---
# ==============================================================================

# 1. 使用固定的 "裁判" LLM，不再是列表
JUDGE_LLM_CONFIG = {
    "name": "Qwen2.5-7B-Instruct-Judge",
    "model_path": "Qwen/Qwen2.5-7B-Instruct",
    "tp_size": 1
}

# 2. 默认输入是文件夹，而非单个文件
DEFAULT_RESULTS_FOLDER = "/home/yijiexu/LLaMA-Factory/results"
# 3. 默认只处理父目录名包含 'tent' 或 'eata' 的文件
DEFAULT_INCLUDE_DIRS = [
    # "tent", "eata", "eata_0", "eata_1", "eata_160", "eata_sdiv", "eata_sdiv_1",
    # "eata_sdiv_40", "eata_sdiv_160", "tent_0", "tent_1", "tent_40", "tent_160",
    #"base",
    #"tent_1_fullent",
    "nll_0",
    "ppl_0",
    "nll_nll_0",
    "nll_ppl_0",
    "ppl_nll_0",
    "ppl_ppl_0",
    "eata",
    "sft_0",
    "tent"
]
# 4. 新的输出文件后缀
OUTPUT_SUFFIX = "_LLM_EM.jsonl"
SUMMARY_SUFFIX = "_LLM_EM_summary.csv"  # 【新】摘要文件后缀

# 其他核心配置
SGLANG_HOST = "127.0.0.1"
REQUESTED_SGLANG_PORT = 30000
MAX_INPUT_TEXT_LEN = 5120
MAX_NEW_TOKENS = 5120
MAX_RETRIES = 20
N_REPETITIONS = 1
BATCH_SIZE = 1000  # 推理时的批次大小 (恢复为您的设置)

# ==============================================================================
# --- 核心工具和函数 ---
# ==============================================================================

answer_comparison_tool = {
    "type": "function",
    "function": {
        "name": "record_answer_comparison",
        "description":
        "Records if the final answers from two texts are semantically identical.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step thinking."
                },
                "are_answers_identical": {
                    "type": "string",
                    "description": "'Yes' if identical, 'No' otherwise.",
                    "enum": ["Yes", "No"]
                },
            },
            "required": ["reasoning", "are_answers_identical"],
        },
    },
}


def find_jsonl_files(root_folder: str,
                     include_keywords: list[str]) -> list[Path]:
    """
    遍历指定文件夹, 根据关键词过滤并找到所有 .jsonl 文件。
    """
    found_files = []
    print(f"Searching for .jsonl files in '{root_folder}'...")
    print(f"Filtering for directories containing any of: {include_keywords}")

    for root, dirs, files in os.walk(root_folder):
        if any(keyword in root.split(os.sep) for keyword in include_keywords):
            for file in files:
                if file.endswith(".jsonl") and not file.endswith(
                        OUTPUT_SUFFIX) and file.startswith("rb"):
                    found_files.append(Path(root) / file)

    print(f"Found {len(found_files)} files to process.")
    return found_files


def load_and_prepare_data(file_path: Path) -> list[dict]:
    """加载单个JSONL文件。"""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(
                        f"Warning: Skipping malformed JSON line in {file_path}"
                    )
        return samples
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []


def save_results_to_jsonl(results: list[dict], output_path: Path):
    """
    将处理结果逐行写入新的 JSONL 文件。
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Successfully saved {len(results)} judgements to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")


# --- 【新功能】保存摘要的函数 ---
def save_summary_to_csv(results: list[dict], summary_path: Path):
    """
    计算摘要统计信息并将其保存到 CSV 文件。
    """
    judgements = [
        item["llm_judgement"]["is_match_pred"] for item in results
        if item.get("llm_judgement")
        and item["llm_judgement"].get("is_match_pred") is not None
    ]
    total_samples = len(results)
    valid_judgements = len(judgements)
    match_count = sum(1 for j in judgements if j is True)

    valid_rate = (valid_judgements / total_samples) if total_samples > 0 else 0
    match_rate = (match_count /
                  valid_judgements) if valid_judgements > 0 else 0

    summary_data = {
        "total_samples": [total_samples],
        "valid_judgements": [valid_judgements],
        "valid_rate": [f"{valid_rate:.2%}"],
        "match_count (is_true)": [match_count],
        "match_rate (is_true_rate)": [f"{match_rate:.2%}"]
    }

    try:
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_path, index=False)
        print(f"Successfully saved summary to {summary_path}")
    except Exception as e:
        print(f"Error saving summary to {summary_path}: {e}")


def start_sglang_server_programmatic(model_path, requested_port, tp_size=1):
    """恢复为您提供的版本"""
    cmd = [
        "python", "-m", "sglang.launch_server", "--model-path", model_path,
        "--host", SGLANG_HOST, "--port",
        str(requested_port), "--log-level", "info", "--trust-remote-code",
        "--tool-call-parser", "qwen25", "--chat-template",
        "qwen25_chat_template.jinja", "--enable-mixed-chunk"
    ]
    if tp_size > 1:
        cmd += ["--tp-size", str(tp_size)]
    dp_size = torch.cuda.device_count() // tp_size
    cmd += ["--dp-size", str(dp_size)]

    cmd_str = " ".join(cmd)
    print(f"Starting SGLang server: {cmd_str}")
    try:
        proc, port = launch_server_cmd(cmd_str)
        wait_for_server(f"http://{SGLANG_HOST}:{port}", timeout=600)
        print("Server is ready.")
        return proc, port
    except Exception as e:
        print(f"Failed to start server: {e}")
        if "proc" in locals() and proc:
            terminate_process(proc)
        return None, None


def stop_sglang_server_programmatic(proc):
    if proc:
        print(f"Stopping SGLang server (PID {proc.pid})...")
        terminate_process(proc)
        print("Stop request sent.")
    time.sleep(5)


def parse_and_validate_prediction(resp_json):
    """【最终决定版】解析器，处理多种错误模式。"""
    prediction = {
        "reasoning": "Error: Invalid response structure.",
        "is_match_pred": None
    }
    is_valid = False
    arguments_obj = None
    try:
        msg = resp_json.get("choices", [{}])[0].get("message", {})
        if msg.get("tool_calls"):
            arguments_obj = json.loads(
                msg["tool_calls"][0]["function"]["arguments"])
        else:
            content = msg.get("content", "")
            if content:
                start_index, end_index = content.find('{'), content.rfind('}')
                if start_index != -1 and end_index > start_index:
                    json_str = content[start_index:end_index + 1]
                    try:
                        full_obj = json.loads(json_str)
                        if "arguments" in full_obj and isinstance(
                                full_obj.get("arguments"), dict):
                            arguments_obj = full_obj["arguments"]
                        elif "reasoning" in full_obj and "are_answers_identical" in full_obj:
                            arguments_obj = full_obj
                    except json.JSONDecodeError:
                        pass
        if arguments_obj and isinstance(arguments_obj, dict):
            match_val = arguments_obj.get("are_answers_identical")
            if match_val in ["Yes", "No"]:
                prediction["reasoning"] = arguments_obj.get(
                    "reasoning", "No reasoning provided.")
                prediction["is_match_pred"] = (match_val == "Yes")
                is_valid = True
            else:
                prediction[
                    "reasoning"] = f"Error: Invalid 'are_answers_identical' value ('{match_val}')"
        else:
            prediction[
                "reasoning"] = f"Error: Parsing failed. Raw Content: '{msg.get('content', '')}'"
    except Exception as e:
        prediction[
            "reasoning"] = f"Error: Critical parsing error. Details: {e}"
    return prediction, is_valid


async def process_sample_async(client, api_url, model_id, sample,
                               sample_index):
    """处理单个样本的异步函数。"""
    sys_prompt = (
        "You are an expert evaluator. Your task is to determine if the final answer in a 'Prediction' text is exactly the same as the final answer in a 'Ground Truth' text. "
        "The format can be irregular. You need to understand the text to find the final answers and compare their values. "
        "For example, '1,000' and '1000' are identical. '$12.5' and '12.5 dollars' are identical. '1/2' and '0.5' are identical. "
        "Your final decision must be 'Yes' or 'No'. You MUST use the `record_answer_comparison` function."
    )
    usr_prompt = (
        f"Analyze the following pair:\n\n### Prediction:\n{str(sample.get('predict', ''))[:MAX_INPUT_TEXT_LEN]}\n\n"
        f"### Ground Truth:\n{str(sample.get('label', ''))[:MAX_INPUT_TEXT_LEN]}"
    )
    msgs = [{
        "role": "system",
        "content": sys_prompt
    }, {
        "role": "user",
        "content": usr_prompt
    }]
    request_body = {
        "model": model_id,
        "messages": msgs,
        "tools": [answer_comparison_tool],
        "temperature": 0.0,
        "max_tokens": MAX_NEW_TOKENS
    }

    final_prediction = None
    last_error = "No response after retries."
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(api_url,
                                         json=request_body,
                                         timeout=180.0)
            response.raise_for_status()
            prediction, is_valid = parse_and_validate_prediction(
                response.json())
            if is_valid:
                final_prediction = prediction
                break
            else:
                last_error = prediction.get("reasoning", "Validation failed.")
        except httpx.HTTPStatusError as e:
            last_error = f"API Error: {e} - Body: {e.response.text}"
        except Exception as e:
            last_error = f"General Error: {e}"

        # 恢复为您提供的版本
        # if attempt < MAX_RETRIES - 1:
        #     await asyncio.sleep(1)
        else:
            final_prediction = {
                "reasoning":
                f"[Fatal Error after {MAX_RETRIES} retries] {last_error}",
                "is_match_pred": None,
            }
            break

    output_item = sample.copy()
    output_item["llm_judgement"] = final_prediction if final_prediction else {
        "reasoning": f"[Fatal Error] {last_error}",
        "is_match_pred": None
    }
    return output_item


async def run_batch_inference(model_id, base_url, samples: list[dict],
                              run_desc: str) -> list[dict]:
    """对一个文件的所有样本进行批量推理。"""
    api_url = f"{base_url}/v1/chat/completions"
    results = []
    async with httpx.AsyncClient(timeout=190.0) as client:
        with tqdm_asyncio(total=len(samples), desc=run_desc) as pbar:
            for i in range(0, len(samples), BATCH_SIZE):
                batch_samples = samples[i:i + BATCH_SIZE]
                tasks = [
                    process_sample_async(client, api_url, model_id, s, i + j)
                    for j, s in enumerate(batch_samples)
                ]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                pbar.update(len(batch_samples))
    return results


def calculate_and_print_summary(results: list[dict]):
    """为单个文件的结果打印摘要。"""
    judgements = [
        item["llm_judgement"]["is_match_pred"] for item in results
        if item.get("llm_judgement")
        and item["llm_judgement"].get("is_match_pred") is not None
    ]
    total_samples, valid_judgements = len(results), len(judgements)
    match_count = sum(1 for j in judgements if j is True)

    print("\n--- Judgement Summary ---")
    print(f"  Total Samples Processed: {total_samples}")
    if total_samples > 0:
        print(
            f"  Valid LLM Judgements: {valid_judgements} ({valid_judgements/total_samples:.2%})"
        )
    if valid_judgements > 0:
        print(
            f"  Judged as 'Match' (Yes): {match_count} ({match_count/valid_judgements:.2%})"
        )
        print(
            f"  Judged as 'No Match' (No): {valid_judgements - match_count} ({(valid_judgements - match_count)/valid_judgements:.2%})"
        )
    print("-" * 25)


# ==============================================================================
# --- 主函数 Main Function ---
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=
        "Batch process result files to get LLM-based Exact Match judgements.")
    parser.add_argument("--results_folder",
                        type=str,
                        default=DEFAULT_RESULTS_FOLDER,
                        help="Root folder containing .jsonl files to process.")
    parser.add_argument("--include_dirs",
                        type=str,
                        nargs='+',
                        default=DEFAULT_INCLUDE_DIRS,
                        help="List of keywords to filter directories.")
    parser.add_argument("--overwrite",
                        action='store_true',
                        help="Overwrite existing output files.")
    parser.add_argument(
        "--report-only",
        type=str,
        metavar="OUTPUT_FILE",
        help=
        "If specified, skips inference and generates a summary for a single output file."
    )
    args = parser.parse_args()

    # --- 报告模式 (保持原样) ---
    if args.report_only:
        report_file = Path(args.report_only)
        if not report_file.exists():
            print(f"Error: Report file not found at {report_file}")
            return
        print(f"--- REPORT-ONLY MODE for {report_file.name} ---")
        results = load_and_prepare_data(report_file)
        calculate_and_print_summary(results)
        return

    # --- 批处理模式 ---
    files_to_process = find_jsonl_files(args.results_folder, args.include_dirs)
    if not files_to_process:
        print("No files found to process. Exiting.")
        return

    # --- 启动一次SGLang服务 ---
    print(f"Starting LLM Judge: {JUDGE_LLM_CONFIG['name']}")
    proc, port = start_sglang_server_programmatic(
        JUDGE_LLM_CONFIG["model_path"], REQUESTED_SGLANG_PORT,
        JUDGE_LLM_CONFIG["tp_size"])
    if not proc:
        print("Failed to start SGLang server. Aborting.")
        return

    try:
        base_url = f"http://{SGLANG_HOST}:{port}"
        model_id = requests.get(f"{base_url}/v1/models",
                                timeout=20).json()["data"][0]["id"]
        print(f"Server ready. Model ID: {model_id}\n")

        # --- 遍历所有找到的文件 ---
        for i, file_path in enumerate(files_to_process):
            output_path = file_path.with_name(
                f"{file_path.stem}{OUTPUT_SUFFIX}")
            # 【新】定义摘要文件路径
            summary_path = file_path.with_name(
                f"{file_path.stem}{SUMMARY_SUFFIX}")

            print(
                f"Processing file {i+1}/{len(files_to_process)}: {file_path}")

            if output_path.exists() and not args.overwrite:
                print(
                    f"Output file already exists, skipping. Use --overwrite to re-process.\n -> {output_path}"
                )
                continue

            samples = load_and_prepare_data(file_path)
            if not samples:
                print("No samples found, skipping file.")
                continue

            results = asyncio.run(
                run_batch_inference(model_id, base_url, samples,
                                    file_path.name))
            save_results_to_jsonl(results, output_path)

            calculate_and_print_summary(results)
            # 【新】调用保存摘要的函数
            save_summary_to_csv(results, summary_path)

    except Exception as e:
        print(
            f"\nAn unexpected error occurred during the main processing loop: {e}"
        )
        traceback.print_exc()
    finally:
        # --- 关闭SGLang服务 ---
        stop_sglang_server_programmatic(proc)
        print("\nAll tasks complete. Script finished.")


if __name__ == "__main__":
    main()
