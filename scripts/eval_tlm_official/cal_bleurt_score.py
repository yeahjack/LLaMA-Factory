# scripts/eval_tlm_official/cal_bleurt_score.py

# 计算bleurt指标，值越大越好

from bleurt import score
from tqdm import tqdm
import tensorflow as tf

# --- 新增代码：配置GPU显存按需增长 ---
# 1. 获取所有物理上的GPU设备
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # 2. 遍历所有GPU，并为它们设置显存增长模式
    #    这必须在任何TensorFlow会话或操作开始之前完成
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Enabled memory growth for {len(gpus)} GPU(s).")
  except RuntimeError as e:
    # 显存增长必须在GPU初始化之前设置
    print(f"Could not set memory growth: {e}")

def calculate_bleurt_score(candidate, reference):
    """
    计算答案和标准答案之间的 BLEURT 分数
    :param candidate: 候选答案列表或字符串
    :param reference: 标准输出列表或字符串
    :return: 输出 BLEURT 得分列表，[0,1]，得分越大越好
    """
    from pathlib import Path
    import os

    if isinstance(candidate, str):
        candidate = [candidate]
    if isinstance(reference, str):
        reference = [reference]
    if not isinstance(candidate, list) or not isinstance(reference, list):
        raise TypeError("candidate 和 reference 需要是列表或字符串")
    if len(candidate) != len(reference):
        raise ValueError("candidate 和 reference 的长度需要一致")

    model_name = "bleurt-base-128"
    search_dirs = []
    env_cache = os.getenv("BLEURT_CACHE")
    if env_cache:
        search_dirs.append(Path(env_cache).expanduser() / model_name)
    search_dirs.append(Path.home() / ".bleurt" / model_name)
    search_dirs.append(Path.cwd() / model_name)

    marker_files = ("bleurt_config.json", "saved_model.pb")
    ckpt_path = None
    for d in search_dirs:
        if d.is_dir() and all((d / f).exists() for f in marker_files):
            ckpt_path = str(d)
            break

    if ckpt_path is not None:
        scorer = score.BleurtScorer(ckpt_path)
    else:
        try:
            scorer = score.BleurtScorer(model_name)
        except Exception as e:
            raise FileNotFoundError(
                "未找到 bleurt-base-128 模型目录。请将 checkpoint 解压到 ~/.bleurt/bleurt-base-128，"
                "或设置环境变量 BLEURT_CACHE 指向包含该目录的位置。目录需包含 bleurt_config.json 与 saved_model.pb。"
                f"原始错误：{type(e).__name__}: {e}"
            )

    results = []
    for i in tqdm(range(len(candidate)), desc='Evaluating BLEURT score', leave=False):
        scores = scorer.score(references=[reference[i]], candidates=[candidate[i]])
        results.append(scores[0])
    return results


if __name__ == '__main__':
    candidate_answer = [
        "A skeptic is someone who doubts or expresses doubt about a claim or idea without being dismissive of it. They are open-minded and approach evidence with an open mind, searching for reasonable explanations and evidence to support their beliefs.\n\nA denier, on the other hand, is someone who actively works to deny or ignore evidence that contradicts their beliefs. They are often characterized by a closed mind and an unwillingness to consider alternative perspectives. They may also use rhetoric or false claims to try to discredit the evidence.",
        "Can you explain?\n5. I've also noticed that some people who are skeptical about climate change also tend to be skeptical about other scientific subjects, like evolution. Can you explain that?\n6. What evidence have you seen that supports the theory of evolution?\n\nThese are just a few examples of questions that a journalist might ask to gather additional information about someone's skepticism about climate change. It's important for journalists to do their own research and fact-checking to ensure that their stories are accurate and balanced.",
        "Here are a few definitions that I found online:\nSkeptic: a person who seeks to acquire and validate knowledge by investigation and analysis, especially of a scientific or mathematical nature.\nDenier: a person who deliberately refuses to accept facts or evidence that contradict their beliefs.\nIt looks like a skeptic is someone who is open to looking at evidence and facts, while a denier is someone who actively refuses to accept evidence that contradicts their beliefs. I guess that means a skeptic can be wrong, but a denier will never change their mind.\nI think it's important to keep an open mind when it comes to facts and evidence, so I guess I'm a skeptic. What about you?\nI'm always interested in learning new things, and I love when facts and evidence contradict my own beliefs. That's when I know I'm really learning something!"
    ]
    reference_answer = [
        "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason.",
        "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason.",
        "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason."
    ]
    print(calculate_bleurt_score(candidate_answer, reference_answer))