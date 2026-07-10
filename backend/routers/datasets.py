"""数据集路由：上传文件(docx/txt)、抓取网页、列表、删除"""
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from auth import get_current_user, require_admin
from database import get_db
from models import User, Dataset
from schemas import DatasetUrlRequest, DatasetOut, ApiResponse
from services.dataset_service import save_file_dataset, save_url_dataset
from config import DATASET_DIR

router = APIRouter(prefix="/api/datasets", tags=["数据集"])


@router.get("", response_model=list[DatasetOut])
def list_datasets(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@router.post("/upload", response_model=DatasetOut)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    filename = file.filename or "untitled.txt"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ("docx", "txt", "md", "csv"):
        raise HTTPException(status_code=400, detail="仅支持 docx / txt / md / csv 文件")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")
    name = filename.rsplit(".", 1)[0]
    ds = save_file_dataset(filename, raw, name)
    return DatasetOut.model_validate(ds)


@router.post("/url", response_model=DatasetOut)
def add_url(
    payload: DatasetUrlRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    try:
        ds = save_url_dataset(payload.url, payload.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"抓取网页失败：{e}")
    return DatasetOut.model_validate(ds)


@router.delete("/{dataset_id}", response_model=ApiResponse)
def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    # 清理本地文件
    if ds.file_path:
        try:
            p = Path(ds.file_path)
            if p.exists() and DATASET_DIR in p.resolve().parents:
                p.unlink()
        except Exception:
            pass
    db.delete(ds)
    db.commit()
    return ApiResponse(message="数据集已删除")
