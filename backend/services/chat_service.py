"""真实对话推理服务

基于本地下载的 Qwen / 通用 CausalLM 模型进行推理，使用 Apple MPS（或 CPU 回退）加速。
单例模式加载模型，避免每次对话重新加载。

支持：
- 真实流式生成（按 token 输出）
- 多轮对话上下文（chat template）
- 自动剥离 Qwen3 的 <think>...</think> 思考过程
- 模型懒加载 + 全局缓存
"""
import re
import threading
import torch
from pathlib import Path
from typing import Optional

from config import settings


# 模型缓存：key=model_path，value=dict(model, tokenizer, device)
_MODEL_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _strip_think(text: str) -> str:
    """剥离 Qwen3 思考模型的 <think>...</think> 内容"""
    # 移除完整的 think 块
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # 处理未闭合的 <think> 开头（流式中常见）
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _load_model(model_path: str):
    """加载模型与分词器（带缓存，双重检查锁定避免持锁加载）"""
    # 第一次检查（锁外，快速路径）
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    with _CACHE_LOCK:
        # 第二次检查（锁内，防止重复加载）
        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]

        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = _get_device()
        # MPS 在某些 op 上不稳定，使用 float16；CPU 用 float32
        dtype = torch.float16 if device != "cpu" else torch.float32

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()

        info = {"model": model, "tokenizer": tok, "device": device}
        _MODEL_CACHE[model_path] = info
        return info


def _build_prompt(tokenizer, history: list, user_message: str) -> str:
    """构造对话 prompt：用 chat template 拼接历史"""
    messages = []
    # 历史记录：偶数位置是 user，奇数是 assistant
    for i, content in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    try:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return text
    except Exception:
        # 模型没有 chat template，回退到简单拼接
        parts = []
        for i, content in enumerate(history):
            role = "用户" if i % 2 == 0 else "助手"
            parts.append(f"{role}：{content}")
        parts.append(f"用户：{user_message}\n助手：")
        return "\n".join(parts)


def generate_reply_stream(user_message: str, history: list, model_path: str):
    """真实流式生成：逐 token 返回最终回复（剥离 think）

    yields: 文本块字符串
    """
    info = _load_model(model_path)
    model = info["model"]
    tokenizer = info["tokenizer"]
    device = info["device"]

    prompt = _build_prompt(tokenizer, history, user_message)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # 用 TextIteratorStreamer 实现真流式
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # 在线程中跑生成，主线程读取流
    gen_error = [None]

    def _generate():
        try:
            model.generate(**generate_kwargs)
        except Exception as e:
            gen_error[0] = e

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    # 直接透传所有 token（包含 <think>...</think> 思考过程）
    for raw_chunk in streamer:
        if raw_chunk:
            yield raw_chunk

    thread.join()
    if gen_error[0]:
        raise gen_error[0]


def generate_reply(user_message: str, history: list, model_path: str) -> str:
    """非流式：返回完整回复"""
    chunks = []
    for c in generate_reply_stream(user_message, history, model_path):
        chunks.append(c)
    return "".join(chunks).strip()


# ============== 兼容旧接口（mock 回退） ==============
_RESPONSE_TEMPLATES = [
    "您好！我是由 EaseMind 平台训练的 AI 助手。关于您提到的「{topic}」，我可以从以下几个方面来为您解答：\n\n1. **基本概念**：这是该主题的核心要点。\n2. **应用场景**：它在实际中有很多用途。\n3. **注意事项**：使用时需要关注一些细节。\n\n请问您想深入了解哪一部分？",
    "这是一个很有意思的问题。针对「{topic}」，我的理解如下：\n\n简单来说，它涉及到多个方面的知识。根据相关资料和常见实践，我们可以这样理解其工作原理与价值。\n\n如果您有更具体的问题，欢迎继续提问。",
    "关于「{topic}」，我整理了一些信息供您参考：\n\n- 它是一个重要的概念\n- 在实际应用中需要结合具体场景\n- 建议从基础开始逐步深入\n\n希望这些信息对您有帮助，还有什么我可以为您解答的吗？",
]


def _extract_topic(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(请问|帮我|我想了解|你能告诉我|什么是|怎么样|如何|为什么|请介绍一下)[：:，,]?\s*", "", text)
    text = re.sub(r"[？?。.!！]$", "", text)
    return text[:30] if text else "这个问题"


def _mock_reply(user_message: str, history: list) -> str:
    topic = _extract_topic(user_message)
    idx = len(history) % len(_RESPONSE_TEMPLATES)
    return _RESPONSE_TEMPLATES[idx].format(topic=topic)


def _mock_stream(user_message: str, history: list):
    reply = _mock_reply(user_message, history)
    chunks = re.split(r"(\n|。|！|？|，|；|：)", reply)
    buffer = ""
    import time
    for chunk in chunks:
        buffer += chunk
        if chunk in ("\n", "。", "！", "？", "，", "；", "：") or len(buffer) >= 12:
            yield buffer
            buffer = ""
            time.sleep(0.05)
    if buffer:
        yield buffer


def stream_reply(user_message: str, history: list, model_path: Optional[str] = None):
    """统一入口：优先真实推理，model_path 为空或失败时回退 mock"""
    # 配置为 mock 模式
    if settings.TRAINING_MODE.lower() == "mock" and not model_path:
        yield from _mock_stream(user_message, history)
        return

    # 没提供模型路径
    if not model_path:
        yield "（未指定可用模型，无法进行真实推理）"
        return

    # 路径不存在
    if not Path(model_path).exists():
        yield f"（模型路径不存在：{model_path}）"
        return

    try:
        yield from generate_reply_stream(user_message, history, model_path)
    except Exception as e:
        # 真实推理失败，回退到 mock
        yield f"\n\n[推理错误，已回退到模拟回复：{type(e).__name__}: {e}]\n\n"
        yield from _mock_stream(user_message, history)
