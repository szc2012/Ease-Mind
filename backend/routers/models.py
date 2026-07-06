"""模型管理路由：下载魔搭模型、列表、设为可用、删除"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from auth import get_current_user, require_admin
from database import get_db
from models import User, AIModel
from schemas import ModelDownloadRequest, ModelUpdateRequest, ModelOut, ApiResponse
from services.modelscope_service import start_download, read_log

router = APIRouter(prefix="/api/models", tags=["模型管理"])


@router.get("", response_model=list[ModelOut])
def list_models(
    source: str | None = Query(None, description="按来源过滤：modelscope/finetune"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(AIModel)
    if source:
        q = q.filter(AIModel.source == source)
    return q.order_by(AIModel.created_at.desc()).all()


@router.post("/download", response_model=ApiResponse)
def download_model(
    payload: ModelDownloadRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    model_id = payload.model_id.strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="模型ID不能为空")
    name = payload.name or model_id.split("/")[-1]
    model = AIModel(
        name=name,
        source="modelscope",
        model_id=model_id,
        status="pending",
        description=f"从魔搭社区下载：{model_id}",
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    start_download(model)
    return ApiResponse(success=True, message="已开始下载模型，请查看日志", data={"id": model.id})


@router.get("/{model_id}/log", response_model=ApiResponse)
def model_log(
    model_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    model = db.query(AIModel).filter(AIModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    return ApiResponse(data={"log": read_log(model.id), "status": model.status})


@router.patch("/{model_id}", response_model=ModelOut)
def update_model(
    model_id: str,
    payload: ModelUpdateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    model = db.query(AIModel).filter(AIModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    if payload.is_active:
        # 设为可用时，取消其他模型的可用状态
        db.query(AIModel).filter(AIModel.is_active == True).update({AIModel.is_active: False})
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(model, k, v)
    db.commit()
    db.refresh(model)
    return model


@router.delete("/{model_id}", response_model=ApiResponse)
def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    model = db.query(AIModel).filter(AIModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    db.delete(model)
    db.commit()
    return ApiResponse(message="模型已删除")
