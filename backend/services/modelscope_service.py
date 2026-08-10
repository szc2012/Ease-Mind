"""ModelScope 模型下载服务：使用 modelscope SDK 真实下载。

若 SDK 未安装或下载异常，会记录详细错误到日志，并将状态置为 failed。
"""
import os
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

from config import MODEL_DIR, LOG_DIR
from database import SessionLocal
from models import AIModel

# 尝试导入 modelscope SDK
try:
    from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
    HAS_MODELSCOPE = True
except Exception:
    try:
        from modelscope import snapshot_download  # type: ignore
        HAS_MODELSCOPE = True
    except Exception:
        HAS_MODELSCOPE = False


def _log_path(model_id: str) -> Path:
    return LOG_DIR / f"model_{model_id}.log"


def _read_config(local_path) -> dict:
    """读取模型目录下的 config.json，失败返回空 dict"""
    try:
        cfg_file = Path(local_path) / "config.json"
        if cfg_file.exists():
            import json as _json
            return _json.loads(cfg_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _is_causal_lm(cfg: dict) -> bool:
    """判断是否为文本生成（CausalLM）模型。

    依据 architectures 与 model_type 排除语音(ASR)/音频/视觉/OCR/TTS 等
    非文本生成架构，这些模型无法用于本平台的文本训练与对话。
    """
    arch = ((cfg.get("architectures") or [""])[0]).lower()
    mtype = (cfg.get("model_type") or "").lower()
    text = arch + " " + mtype
    # 明确不支持的类型关键词（纯语音/音频/视觉/OCR/TTS 等，无法做文本生成）
    blocked = ["asr", "whisper", "speech", "tts", "musicgen", "ocr", "clip", "blip"]
    if any(kw in text for kw in blocked):
        return False
    # 统一多模态模型（如 gemma4_unified）支持文本生成 → 放行，走文本模式
    if "unified" in mtype or "unified" in arch:
        return True
    # ForConditionalGeneration 但不带 Causal/LM/Unified 字样 → 多为 seq2seq/纯多模态，拒绝
    if "forconditionalgeneration" in arch and "causal" not in arch and "lm" not in arch:
        return False
    return True


def _log(model_id: str, message: str) -> None:
    """追加一行日志"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    with open(_log_path(model_id), "a", encoding="utf-8") as f:
        f.write(line)


def _set_status(model_id: str, status: str, **fields) -> None:
    """更新模型状态"""
    db = SessionLocal()
    try:
        m = db.query(AIModel).filter(AIModel.id == model_id).first()
        if not m:
            return
        m.status = status
        for k, v in fields.items():
            setattr(m, k, v)
        m.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def _download_real(model_id: str, model_repo: str) -> None:
    """使用 modelscope SDK 真实下载"""
    _log(model_id, f"开始从魔搭社区下载模型：{model_repo}")
    if not HAS_MODELSCOPE:
        _log(model_id, "[错误] 未安装 modelscope SDK，无法真实下载。")
        _log(model_id, "请执行：pip install modelscope")
        _set_status(model_id, "failed", description="未安装 modelscope SDK")
        return

    _set_status(model_id, "downloading")
    _log(model_id, "已检测到 modelscope SDK，开始真实下载...")

    # 校验 model_repo 格式，防止路径异常
    import re
    if not re.match(r"^[a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+$", model_repo):
        _set_status(model_id, "failed", description=f"模型 ID 格式非法：{model_repo}")
        _log(model_id, f"[错误] 模型 ID 格式非法：{model_repo}（应为 org/name 格式）")
        return

    # 明确目标目录：MODEL_DIR / 模型名(转下划线)
    target_name = model_repo.replace("/", "_")
    target_dir = MODEL_DIR / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    _log(model_id, f"本地存储目录：{target_dir}")

    try:
        # 调用 SDK 下载（revision=None 表示下载主干；cache_dir 指定缓存位置）
        _log(model_id, "调用 snapshot_download()，这可能需要一段时间（取决于模型大小与网络）...")
        start_ts = time.time()
        local_path = snapshot_download(
            model_repo,
            cache_dir=str(MODEL_DIR),
        )
        elapsed = round(time.time() - start_ts, 1)

        # SDK 返回的可能是真实路径，也可能是缓存子路径；统一为 Path
        local_path = Path(local_path)
        _log(model_id, f"SDK 返回路径：{local_path}（耗时 {elapsed}s）")

        # 校验下载结果
        if not local_path.exists():
            _log(model_id, f"[警告] 返回路径不存在：{local_path}，尝试在 cache 中查找...")
            # modelscope 通常缓存到 MODEL_DIR/<org>_<name>/ 或 MODEL_DIR/<org>/<name>/
            candidates = [
                MODEL_DIR / target_name,
                MODEL_DIR / model_repo.split("/")[0] / model_repo.split("/")[-1],
            ]
            for c in candidates:
                if c.exists() and any(c.iterdir()):
                    local_path = c
                    _log(model_id, f"找到缓存目录：{local_path}")
                    break

        files = list(local_path.iterdir()) if local_path.exists() else []
        file_names = [f.name for f in files]
        _log(model_id, f"下载文件列表（{len(files)} 项）：{', '.join(file_names[:10])}{'...' if len(file_names) > 10 else ''}")

        # 校验是否包含必要文件
        has_config = any("config" in n.lower() for n in file_names)
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        size_mb = round(total_size / 1024 / 1024, 2)
        _log(model_id, f"总大小：{size_mb} MB")

        if not has_config and len(files) == 0:
            _log(model_id, "[错误] 下载结果为空目录，可能模型名错误或下载被中断。")
            _set_status(model_id, "failed", description="下载结果为空")
            return

        # 校验是否为文本生成（CausalLM）模型，拒绝 ASR/语音/视觉等不支持的类型
        cfg = _read_config(local_path)
        arch = (cfg.get("architectures") or [""])[0]
        mtype = cfg.get("model_type", "")
        if cfg and not _is_causal_lm(cfg):
            _log(model_id, f"[错误] 模型类型不支持文本训练/对话：model_type={mtype}, architecture={arch}")
            _log(model_id, "仅支持文本生成类模型（如 Qwen3-0.6B、minimind 等 CausalLM）。")
            _set_status(model_id, "failed",
                        description=f"不支持的模型类型（{mtype or arch}），仅支持文本生成模型")
            return

        _log(model_id, "=" * 50)
        _log(model_id, "模型下载完成，已就绪！")
        _log(model_id, "=" * 50)
        _set_status(
            model_id,
            "ready",
            local_path=str(local_path),
            description=f"从魔搭社区下载：{model_repo}（{size_mb} MB）",
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        _log(model_id, "[错误] 下载失败：" + err)
        _log(model_id, "详细堆栈：\n" + tb)
        _set_status(model_id, "failed", description=f"下载失败：{err}")
        _log(model_id, "=" * 50)
        _log(model_id, "提示：请检查")
        _log(model_id, "  1. 模型名是否正确（例：Qwen/Qwen3-0.6B）")
        _log(model_id, "  2. 网络是否能访问 modelscope.cn")
        _log(model_id, "  3. 磁盘空间是否充足")


def start_download(model: AIModel) -> None:
    """异步启动模型下载线程（仅真实下载；SDK 未装时直接标记失败）"""
    if not HAS_MODELSCOPE:
        _log(model.id, "[错误] 未安装 modelscope SDK，无法下载。请执行：pip install modelscope")
        _set_status(model.id, "failed", description="未安装 modelscope SDK")
        return
    t = threading.Thread(
        target=_download_real,
        args=(model.id, model.model_id),
        daemon=True,
    )
    t.start()


def read_log(model_id: str) -> str:
    p = _log_path(model_id)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")
