"""ModelScope 模型下载服务

根据 settings.MODEL_DOWNLOAD_MODE 决定下载方式：
- "real"    使用 modelscope SDK 真实下载（默认）
- "mock"    模拟下载（不联网，仅生成占位文件，用于演示流程）

真实模式下若 SDK 未安装或下载异常，会记录详细错误到日志，并将状态置为 failed。
"""
import os
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

from config import MODEL_DIR, LOG_DIR, settings
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


def _download_mock(model_id: str, model_repo: str) -> None:
    """模拟下载进度，便于无网络/无 SDK 环境体验完整流程"""
    _set_status(model_id, "downloading")
    _log(model_id, f"（模拟模式）开始下载模型：{model_repo}")
    steps = [
        "解析模型仓库信息...",
        "获取文件列表：config.json, model.safetensors, tokenizer.json",
        "下载 config.json (1.2 KB) ✓",
        "下载 tokenizer.json (876 KB) ✓",
        "下载 model.safetensors (4.2 GB)...",
        "下载 model.safetensors 进度 25%...",
        "下载 model.safetensors 进度 50%...",
        "下载 model.safetensors 进度 75%...",
        "下载 model.safetensors 进度 100% ✓",
        "校验文件完整性...",
        "模型加载就绪。",
    ]
    for step in steps:
        time.sleep(0.8)
        _log(model_id, step)
    fake_path = MODEL_DIR / model_repo.replace("/", "_")
    fake_path.mkdir(parents=True, exist_ok=True)
    (fake_path / "config.json").write_text('{"mock": true}', encoding="utf-8")
    _log(model_id, f"（模拟）模型下载完成：{fake_path}")
    _log(model_id, "注意：这是模拟模式，仅生成占位文件。若需真实下载，请在 config.py 设置 MODEL_DOWNLOAD_MODE='real'")
    _set_status(
        model_id,
        "ready",
        local_path=str(fake_path),
        description=f"（模拟）从魔搭社区下载：{model_repo}",
    )


def start_download(model: AIModel) -> None:
    """异步启动模型下载线程"""
    mode = settings.MODEL_DOWNLOAD_MODE.lower()
    # 真实模式但 SDK 未装时，自动降级为 mock 并提示
    if mode == "real" and not HAS_MODELSCOPE:
        _log(model.id, "[警告] MODEL_DOWNLOAD_MODE='real' 但未安装 modelscope SDK，本次将使用模拟模式。")
        _log(model.id, "请执行：pip install modelscope")
        target = _download_mock
    elif mode == "mock":
        target = _download_mock
    else:
        target = _download_real

    t = threading.Thread(
        target=target,
        args=(model.id, model.model_id),
        daemon=True,
    )
    t.start()


def read_log(model_id: str) -> str:
    p = _log_path(model_id)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")
