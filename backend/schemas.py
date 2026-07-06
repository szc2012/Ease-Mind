"""Pydantic 请求/响应模型"""
from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field


# ---- 认证 ----
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=64)


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: str
    username: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---- 模型 ----
class ModelDownloadRequest(BaseModel):
    model_id: str = Field(..., description="魔搭社区模型ID，例如 ZhipuAI/glm-4-9b-chat")
    name: Optional[str] = None


class ModelUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ModelOut(BaseModel):
    id: str
    name: str
    source: str
    model_id: Optional[str]
    local_path: Optional[str]
    status: str
    description: str
    is_active: bool
    base_model_id: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ---- 数据集 ----
class DatasetUrlRequest(BaseModel):
    url: str
    name: Optional[str] = None


class DatasetOut(BaseModel):
    id: str
    name: str
    source_type: str
    source_info: Optional[str]
    sample_count: int
    char_count: int
    content_preview: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---- 训练 ----
class TrainingCreateRequest(BaseModel):
    name: str
    model_id: str
    dataset_id: str
    mode: str = "simple"  # simple / professional
    params: dict = Field(default_factory=dict)


class TrainingOut(BaseModel):
    id: str
    name: str
    model_id: str
    dataset_id: str
    mode: str
    status: str
    progress: float
    params: dict
    result_model_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---- 对话 ----
class ChatSessionCreate(BaseModel):
    model_id: str
    title: Optional[str] = "新对话"


class ChatSessionOut(BaseModel):
    id: str
    model_id: str
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatMessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSendRequest(BaseModel):
    session_id: str
    content: str


class ApiResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: Optional[Any] = None
