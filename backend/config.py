"""应用配置"""
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
DATASET_DIR = DATA_DIR / "datasets"
LOG_DIR = DATA_DIR / "logs"


class Settings(BaseSettings):
    APP_NAME: str = "EaseMind AI 训练平台"
    SECRET_KEY: str = "easemind-secret-key-change-in-production-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'easemind.db'}"

    # 训练模式：mock=模拟（无需GPU），real=真实 LoRA 微调
    TRAINING_MODE: str = "real"

    # 模型下载模式：real=使用 modelscope SDK 真实下载，mock=模拟（仅占位文件）
    MODEL_DOWNLOAD_MODE: str = "real"

    # 默认管理员账户
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"

    HOST: str = "0.0.0.0"
    PORT: int = 8080

    # 调试模式：开启后暴露 /docs，关闭则隐藏 API 文档
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()

# 确保数据目录存在
for d in (DATA_DIR, MODEL_DIR, DATASET_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
