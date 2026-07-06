"""SQLAlchemy 数据模型"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, JSON

from database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(16), default="user", nullable=False)  # admin / user
    created_at = Column(DateTime, default=datetime.utcnow)


class AIModel(Base):
    """基础模型 / 微调后的模型"""
    __tablename__ = "ai_models"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(128), nullable=False)
    source = Column(String(32), default="modelscope")  # modelscope / finetune
    model_id = Column(String(256))  # 魔搭模型ID，如 ZhipuAI/glm-4-9b-chat
    local_path = Column(String(512))
    status = Column(String(32), default="pending")  # pending/downloading/ready/failed
    description = Column(Text, default="")
    is_active = Column(Boolean, default=False)  # 是否为当前对话可用模型
    base_model_id = Column(String(64), nullable=True)  # 微调模型的基础模型
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(128), nullable=False)
    source_type = Column(String(32), nullable=False)  # file / url
    source_info = Column(String(512))  # 文件名或URL
    file_path = Column(String(512))
    sample_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    content_preview = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingTask(Base):
    __tablename__ = "training_tasks"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(128), nullable=False)
    model_id = Column(String(64), nullable=False)
    # 逗号分隔的多个数据集 ID（兼容旧的单 ID 数据）
    dataset_id = Column(String(256), nullable=False)
    mode = Column(String(32), default="simple")  # simple(傻瓜) / professional(专业)
    status = Column(String(32), default="pending")  # pending/running/completed/failed/cancelled
    progress = Column(Float, default=0.0)
    params = Column(JSON, default=dict)
    result_model_id = Column(String(64), nullable=True)
    log_file = Column(String(512), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    @property
    def dataset_ids(self) -> list:
        """解析 dataset_id 字段为 ID 列表"""
        return [s.strip() for s in (self.dataset_id or "").split(",") if s.strip()]


class DistillationTask(Base):
    """模型蒸馏任务：用教师模型蒸馏学生模型"""
    __tablename__ = "distillation_tasks"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(128), nullable=False)
    teacher_model_id = Column(String(64), nullable=False)
    student_model_id = Column(String(64), nullable=False)
    # 逗号分隔的多个数据集 ID
    dataset_id = Column(String(256), nullable=False)
    status = Column(String(32), default="pending")  # pending/running/completed/failed
    progress = Column(Float, default=0.0)
    params = Column(JSON, default=dict)
    result_model_id = Column(String(64), nullable=True)
    log_file = Column(String(512), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    @property
    def dataset_ids(self) -> list:
        return [s.strip() for s in (self.dataset_id or "").split(",") if s.strip()]


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String(64), index=True, nullable=False)
    model_id = Column(String(64), nullable=False)
    title = Column(String(256), default="新对话")
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String(64), index=True, nullable=False)
    role = Column(String(16), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
