"""训练后自动评估服务

用一组预设测试 prompt 分别调用基础模型与微调模型生成回复，
对比两者输出，便于直观判断微调效果。

- 仅在 real 训练模式下触发（mock 模式无真实模型可推理）
- 评估失败不影响训练结果，错误会记录到评估条目的 error 字段
"""
from pathlib import Path
from typing import Optional

from services.chat_service import generate_reply


# 预设评估 prompt：覆盖介绍、知识问答、指令遵循、多轮上下文等场景
EVAL_PROMPTS = [
    "你好，请介绍一下你自己。",
    "请用三句话解释什么是机器学习。",
    "帮我写一段 Python 函数，判断一个字符串是否是回文。",
    "请列举三种常见的排序算法并说明其时间复杂度。",
]


def _safe_generate(prompt: str, model_path: Optional[str], history: list = None) -> str:
    """安全调用模型生成回复，异常时返回错误标记文本"""
    if not model_path or not Path(model_path).exists():
        return f"[评估跳过] 模型路径无效：{model_path}"
    try:
        return generate_reply(prompt, history or [], model_path)
    except Exception as e:
        return f"[推理错误] {type(e).__name__}: {e}"


def evaluate_finetuned_model(
    base_model_path: Optional[str],
    finetuned_model_path: Optional[str],
    prompts: Optional[list[str]] = None,
    max_new_tokens: int = 256,
) -> list[dict]:
    """对基础模型与微调模型跑同一组 prompt，返回对比结果

    返回结构：
        [
            {
                "prompt": "...",
                "base_reply": "...",
                "finetuned_reply": "...",
                "error": null  # 或字符串
            },
            ...
        ]
    """
    eval_prompts = prompts if prompts is not None else EVAL_PROMPTS
    results: list[dict] = []
    for prompt in eval_prompts:
        item = {"prompt": prompt, "base_reply": "", "finetuned_reply": "", "error": None}
        try:
            item["base_reply"] = _safe_generate(prompt, base_model_path)
        except Exception as e:
            item["error"] = f"基础模型推理异常：{e}"
        try:
            item["finetuned_reply"] = _safe_generate(prompt, finetuned_model_path)
        except Exception as e:
            item["error"] = (item["error"] or "") + f"微调模型推理异常：{e}"
        # 截断过长回复，避免数据库膨胀
        if len(item["base_reply"]) > max_new_tokens * 4:
            item["base_reply"] = item["base_reply"][: max_new_tokens * 4] + "..."
        if len(item["finetuned_reply"]) > max_new_tokens * 4:
            item["finetuned_reply"] = item["finetuned_reply"][: max_new_tokens * 4] + "..."
        results.append(item)
    return results
