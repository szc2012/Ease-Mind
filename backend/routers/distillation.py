"""模型蒸馏路由：创建蒸馏任务、列表、详情、日志(SSE流)、参数配置

所有接口仅管理员可用。
"""
import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from auth import get_current_user, require_admin
from database import get_db, SessionLocal
from models import User, DistillationTask, AIModel, Dataset
from schemas import DistillationCreateRequest, DistillationOut, ApiResponse
from services.distillation_service import (
    start_distillation, read_log, get_log_path,
    DISTILL_PARAMS_CONFIG, DISTILL_PARAMS_DEFAULTS,
)
from services.loss_service import get_loss_points

router = APIRouter(prefix="/api/distillation", tags=["模型蒸馏"])


@router.get("/params", response_model=ApiResponse)
def get_param_config(user: User = Depends(require_admin)):
    """返回蒸馏参数配置与默认值"""
    return ApiResponse(data={
        "params_config": DISTILL_PARAMS_CONFIG,
        "defaults": DISTILL_PARAMS_DEFAULTS,
    })


@router.post("", response_model=DistillationOut)
def create_task(
    payload: DistillationCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    teacher = db.query(AIModel).filter(AIModel.id == payload.teacher_model_id).first()
    if not teacher:
        raise HTTPException(status_code=400, detail="教师模型不存在")
    if teacher.status != "ready":
        raise HTTPException(status_code=400, detail="教师模型未就绪")
    student = db.query(AIModel).filter(AIModel.id == payload.student_model_id).first()
    if not student:
        raise HTTPException(status_code=400, detail="学生模型不存在")
    if student.status != "ready":
        raise HTTPException(status_code=400, detail="学生模型未就绪")
    if teacher.id == student.id:
        raise HTTPException(status_code=400, detail="教师模型与学生模型不能相同")

    ids = [s.strip() for s in payload.dataset_ids if s and s.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="请至少选择一个数据集")
    for did in ids:
        if not db.query(Dataset).filter(Dataset.id == did).first():
            raise HTTPException(status_code=400, detail=f"数据集不存在：{did}")

    # 合并参数（默认值 + 用户传入）
    params = dict(DISTILL_PARAMS_DEFAULTS)
    params.update(payload.params or {})

    task = DistillationTask(
        name=payload.name,
        teacher_model_id=payload.teacher_model_id,
        student_model_id=payload.student_model_id,
        dataset_id=",".join(ids),
        status="pending",
        params=params,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    start_distillation(task.id)
    return task


@router.get("", response_model=list[DistillationOut])
def list_tasks(
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    return db.query(DistillationTask).order_by(DistillationTask.created_at.desc()).all()


@router.get("/{task_id}", response_model=DistillationOut)
def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    task = db.query(DistillationTask).filter(DistillationTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="蒸馏任务不存在")
    return task


@router.get("/{task_id}/log", response_model=ApiResponse)
def get_task_log(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    task = db.query(DistillationTask).filter(DistillationTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="蒸馏任务不存在")
    return ApiResponse(data={"log": read_log(task_id), "status": task.status, "progress": task.progress})


@router.get("/{task_id}/log/stream")
async def stream_task_log(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    """SSE 流式推送蒸馏日志"""
    task = db.query(DistillationTask).filter(DistillationTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="蒸馏任务不存在")

    log_path = get_log_path(task_id)
    sent = 0

    async def event_gen():
        nonlocal sent
        last_status = None
        last_progress = None
        tick = 0
        while True:
            if log_path.exists():
                content = log_path.read_text(encoding="utf-8")
                if len(content) > sent:
                    chunk = content[sent:]
                    sent = len(content)
                    yield f"data: {json.dumps({'type': 'log', 'content': chunk})}\n\n"
            db2 = SessionLocal()
            try:
                fresh = db2.query(DistillationTask).filter(DistillationTask.id == task_id).first()
            finally:
                db2.close()
            if fresh:
                progress_delta = abs(fresh.progress - (last_progress or 0))
                if fresh.status != last_status or progress_delta >= 0.5:
                    last_status = fresh.status
                    last_progress = fresh.progress
                    yield f"data: {json.dumps({'type': 'status', 'status': fresh.status, 'progress': fresh.progress})}\n\n"
                if fresh.status in ("completed", "failed"):
                    yield f"data: {json.dumps({'type': 'done', 'status': fresh.status, 'progress': fresh.progress})}\n\n"
                    break
            else:
                # 任务已被删除，退出循环
                yield f"data: {json.dumps({'type': 'done', 'status': 'cancelled', 'progress': 0})}\n\n"
                break
            tick += 1
            if tick % 15 == 0:
                yield f": keep-alive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/{task_id}/loss", response_model=ApiResponse)
def get_task_loss(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    """获取蒸馏任务的 loss 曲线数据"""
    task = db.query(DistillationTask).filter(DistillationTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="蒸馏任务不存在")
    return ApiResponse(data={"points": get_loss_points(task_id, "distillation")})
