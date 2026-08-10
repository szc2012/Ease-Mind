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
import json
import threading
import torch
from pathlib import Path
from typing import Optional




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


def _is_multimodal_unified(model_path: str) -> bool:
    """判断是否为统一多模态模型（如 gemma4_unified），需用多模态类加载以兼容纯文本"""
    try:
        cfg_file = Path(model_path) / "config.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            mtype = (cfg.get("model_type") or "").lower()
            arch = ((cfg.get("architectures") or [""])[0]).lower()
            return "unified" in mtype or "unified" in arch
    except Exception:
        pass
    return False


def _load_model(model_path: str):
    """加载模型与分词器（带缓存，双重检查锁定避免持锁加载）

    文本模式适配：统一多模态模型（gemma4_unified 等）用 AutoModelForMultimodalLM
    加载以兼容纯文本推理；普通文本模型用 AutoModelForCausalLM。
    """
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

        if _is_multimodal_unified(model_path):
            # 统一多模态模型：仅用其文本生成能力
            from transformers import AutoModelForMultimodalLM
            model = AutoModelForMultimodalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
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


def stream_reply(user_message: str, history: list, model_path: Optional[str] = None):
    """统一入口：仅真实推理，失败时返回错误提示"""
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
        yield f"\n\n[推理错误：{type(e).__name__}: {e}]\n\n"
