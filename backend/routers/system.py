"""系统状态路由：GPU 显存、磁盘占用、活跃任务数

仅管理员可用。
"""
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth import require_admin
from config import MODEL_DIR, DATASET_DIR, LOG_DIR, DATA_DIR
from database import get_db
from models import User, TrainingTask, DistillationTask
from schemas import ApiResponse

router = APIRouter(prefix="/api/system", tags=["系统"])


def _dir_size(path: Path) -> int:
    """递归计算目录总字节数（不存在返回 0）"""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _human_bytes(n: int) -> str:
    """字节数转人类可读字符串"""
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    f = n / 1024.0
    for u in units:
        if f < 1024.0:
            return f"{f:.1f} {u}"
        f /= 1024.0
    return f"{f:.1f} PB"


def _gpu_status() -> dict:
    """获取 GPU 显存状态"""
    try:
        import torch
    except Exception:
        return {"available": False, "device": "none", "detail": "torch 未安装"}

    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory
            allocated = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            return {
                "available": True,
                "device": "cuda",
                "name": name,
                "total": total,
                "allocated": allocated,
                "reserved": reserved,
                "total_human": _human_bytes(total),
                "allocated_human": _human_bytes(allocated),
                "reserved_human": _human_bytes(reserved),
            }
        except Exception as e:
            return {"available": True, "device": "cuda", "detail": f"显存查询失败：{e}"}

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS 不暴露显存查询 API，只能提示可用
        return {
            "available": True,
            "device": "mps",
            "name": "Apple Silicon GPU",
            "detail": "MPS 不支持显存精确查询，已统一计入进程内存",
        }

    return {"available": False, "device": "cpu", "detail": "无可用 GPU（仅 CPU）"}


@router.get("/status", response_model=ApiResponse)
def get_system_status(
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    """返回系统资源状态：GPU 显存、磁盘占用、活跃任务数"""
    # 活跃任务数
    active_training = db.query(TrainingTask).filter(TrainingTask.status == "running").count()
    active_distill = db.query(DistillationTask).filter(DistillationTask.status == "running").count()
    pending_training = db.query(TrainingTask).filter(TrainingTask.status == "pending").count()
    pending_distill = db.query(DistillationTask).filter(DistillationTask.status == "pending").count()

    # 各目录磁盘占用
    model_size = _dir_size(MODEL_DIR)
    dataset_size = _dir_size(DATASET_DIR)
    log_size = _dir_size(LOG_DIR)
    db_path = DATA_DIR / "easemind.db"
    db_size = db_path.stat().st_size if db_path.exists() else 0
    total_data_size = model_size + dataset_size + log_size + db_size

    # 数据目录所在分区的整体磁盘使用
    try:
        du = shutil.disk_usage(str(DATA_DIR))
        disk_total = du.total
        disk_used = du.used
        disk_free = du.free
    except Exception:
        disk_total = disk_used = disk_free = 0

    return ApiResponse(data={
        "gpu": _gpu_status(),
        "active_tasks": {
            "running": active_training + active_distill,
            "pending": pending_training + pending_distill,
            "training": active_training,
            "distillation": active_distill,
        },
        "disk": {
            "models": {"bytes": model_size, "human": _human_bytes(model_size)},
            "datasets": {"bytes": dataset_size, "human": _human_bytes(dataset_size)},
            "logs": {"bytes": log_size, "human": _human_bytes(log_size)},
            "database": {"bytes": db_size, "human": _human_bytes(db_size)},
            "data_total": {"bytes": total_data_size, "human": _human_bytes(total_data_size)},
            "partition": {
                "total": disk_total,
                "used": disk_used,
                "free": disk_free,
                "total_human": _human_bytes(disk_total),
                "used_human": _human_bytes(disk_used),
                "free_human": _human_bytes(disk_free),
            },
        },
    })
