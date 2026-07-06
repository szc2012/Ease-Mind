"""Loss 曲线数据服务

将训练/蒸馏过程的 loss 数据点以 JSON 文件形式持久化，供前端绘制曲线。
每个任务一个 JSON 文件，格式：[{step, loss, epoch, lr, ts}, ...]
"""
import json
from pathlib import Path
from datetime import datetime

from config import LOG_DIR


def _loss_file(task_id: str, task_type: str = "training") -> Path:
    """task_type: training / distillation"""
    return LOG_DIR / f"{task_type}_loss_{task_id}.json"


def record_loss(task_id: str, step: int, loss: float, epoch: int = 0,
                lr: float = 0.0, extra: dict = None, task_type: str = "training") -> None:
    """记录一个 loss 数据点"""
    path = _loss_file(task_id, task_type)
    points = []
    if path.exists():
        try:
            points = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            points = []
    point = {
        "step": step,
        "loss": round(float(loss), 6),
        "epoch": epoch,
        "lr": round(float(lr), 8) if lr else 0.0,
        "ts": datetime.now().strftime("%H:%M:%S"),
    }
    if extra:
        for k, v in extra.items():
            try:
                point[k] = round(float(v), 6) if isinstance(v, (int, float)) else v
            except (TypeError, ValueError):
                point[k] = v
    points.append(point)
    path.write_text(json.dumps(points, ensure_ascii=False), encoding="utf-8")


def get_loss_points(task_id: str, task_type: str = "training") -> list:
    """获取全部 loss 数据点"""
    path = _loss_file(task_id, task_type)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []


def clear_loss_points(task_id: str, task_type: str = "training") -> None:
    """清除 loss 数据"""
    path = _loss_file(task_id, task_type)
    if path.exists():
        path.unlink()
