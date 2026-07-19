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
    model_id: str = Field(..., description="魔搭社区模型ID，例如 ZhipusAI/glm-4-9b-chat")
    name: Optional[str] = None


class LocalModelImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="模型显示名称")
    local_path: str = Field(..., description="本地模型目录的绝对路径")
    description: Optional[str] = None


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
    # 支持多数据集训练；兼容旧字段 dataset_id
    dataset_ids: Optional[list[str]] = None
    dataset_id: Optional[str] = None
    mode: str = "simple"  # simple / professional
    params: dict = Field(default_factory=dict)


class TrainingOut(BaseModel):
    id: str
    name: str
    model_id: str
    dataset_id: str
    dataset_ids: list[str] = []
    mode: str
    status: str
    progress: float
    params: dict
    result_model_id: Optional[str]
    error_message: Optional[str]
    evaluation: Optional[list] = None
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---- 模型蒸馏 ----
DISTILL_PARAMS_DEFAULTS = {
    "temperature": 2.0,      # 软化 logits 的温度
    "alpha": 0.5,            # 蒸馏损失权重（0~1）
    "epochs": 2,
    "batch_size": 2,
    "learning_rate": 0.0002,
    "max_seq_length": 256,
}


class DistillationCreateRequest(BaseModel):
    name: str
    teacher_model_id: str
    student_model_id: str
    dataset_ids: list[str]
    params: dict = Field(default_factory=dict)


class DistillationOut(BaseModel):
    id: str
    name: str
    teacher_model_id: str
    student_model_id: str
    dataset_id: str
    dataset_ids: list[str] = []
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


# ---- API Key ----
class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    model_id: Optional[str] = None  # 绑定模型（可选）


class ApiKeyOut(BaseModel):
    id: str
    name: str
    key: str  # 仅创建时返回完整 key，列表只返回前缀
    key_prefix: str = ""
    model_id: Optional[str]
    is_active: bool
    last_used_at: Optional[datetime]
    total_requests: int
    created_at: datetime

    class Config:
        from_attributes = True


# ---- OpenAI 兼容接口 ----
class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "default"
    messages: list[OpenAIChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


# ---- 频道（第三方聊天平台接入）----
class ChannelCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    platform: str = "custom"  # dingtalk / feishu / wecom / custom
    model_id: str
    webhook_url: Optional[str] = None
    secret: Optional[str] = None


class ChannelOut(BaseModel):
    id: str
    name: str
    platform: str
    model_id: str
    model_name: Optional[str] = None
    webhook_url: Optional[str]
    secret: Optional[str]
    session_id: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
