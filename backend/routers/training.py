"""训练路由：创建任务、列表、详情、日志(SSE流)、参数配置"""
import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from auth import get_current_user, require_admin
from config import LOG_DIR
from database import get_db, SessionLocal
from models import User, TrainingTask, AIModel, Dataset
from schemas import TrainingCreateRequest, TrainingOut, ApiResponse
from services.training_service import (
    start_training, read_log, get_log_path,
    PROFESSIONAL_PARAMS, SIMPLE_PRESETS,
)

router = APIRouter(prefix="/api/training", tags=["训练"])


@router.get("/params", response_model=ApiResponse)
def get_param_config(user: User = Depends(get_current_user)):
    """返回傻瓜预设与专业参数配置"""
    return ApiResponse(data={
        "simple_presets": SIMPLE_PRESETS,
        "professional_params": PROFESSIONAL_PARAMS,
    })


@router.post("", response_model=TrainingOut)
def create_task(
    payload: TrainingCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    model = db.query(AIModel).filter(AIModel.id == payload.model_id).first()
    if not model:
        raise HTTPException(status_code=400, detail="基础模型不存在")
    if model.status != "ready":
        raise HTTPException(status_code=400, detail="基础模型未就绪，请等待下载完成")
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=400, detail="数据集不存在")

    params = dict(payload.params)
    if payload.mode == "simple":
        # 傻瓜模式：从预设取参数，补全默认值
        preset_key = params.get("preset", "balanced")
        preset = SIMPLE_PRESETS.get(preset_key, SIMPLE_PRESETS["balanced"])
        params = {"preset": preset_key, "epochs": preset["epochs"], "lora_r": preset["lora_r"],
                  "batch_size": 4, "learning_rate": 0.0002, "lora_alpha": preset["lora_r"] * 2,
                  "lora_dropout": 0.05, "max_seq_length": 512, "warmup_steps": 50, "weight_decay": 0.01}
    else:
        # 专业模式：补全默认值
        for k, cfg in PROFESSIONAL_PARAMS.items():
            if k not in params:
                params[k] = cfg["default"]

    task = TrainingTask(
        name=payload.name,
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        mode=payload.mode,
        status="pending",
        params=params,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    start_training(task.id)
    return task


@router.get("", response_model=list[TrainingOut])
def list_tasks(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return db.query(TrainingTask).order_by(TrainingTask.created_at.desc()).all()


@router.get("/{task_id}", response_model=TrainingOut)
def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@router.get("/{task_id}/log", response_model=ApiResponse)
def get_task_log(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ApiResponse(data={"log": read_log(task_id), "status": task.status, "progress": task.progress})


@router.get("/{task_id}/log/stream")
async def stream_task_log(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """SSE 流式推送训练日志"""
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    log_path = get_log_path(task_id)
    sent = 0

    async def event_gen():
        nonlocal sent
        last_status = None
        last_progress = None
        tick = 0
        while True:
            # 读取新增日志
            if log_path.exists():
                content = log_path.read_text(encoding="utf-8")
                if len(content) > sent:
                    chunk = content[sent:]
                    sent = len(content)
                    yield f"data: {json.dumps({'type': 'log', 'content': chunk})}\n\n"
            # 用独立 session 查询最新状态（避免当前请求 session 缓存）
            db2 = SessionLocal()
            try:
                fresh = db2.query(TrainingTask).filter(TrainingTask.id == task_id).first()
            finally:
                db2.close()
            if fresh:
                # 状态变化 或 进度变化（>0.5%）都推送
                progress_delta = abs(fresh.progress - (last_progress or 0))
                if fresh.status != last_status or progress_delta >= 0.5:
                    last_status = fresh.status
                    last_progress = fresh.progress
                    yield f"data: {json.dumps({'type': 'status', 'status': fresh.status, 'progress': fresh.progress})}\n\n"
                if fresh.status in ("completed", "failed", "cancelled"):
                    yield f"data: {json.dumps({'type': 'done', 'status': fresh.status, 'progress': fresh.progress})}\n\n"
                    break
            # 心跳：每 15 秒发一次 keep-alive，避免代理/浏览器断流
            tick += 1
            if tick % 15 == 0:
                yield f": keep-alive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
