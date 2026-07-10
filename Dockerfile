FROM python:3.12-slim

LABEL maintainer="szc2012"
LABEL description="EaseMind — 零门槛 AI 训练与微调平台"

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先拷贝依赖文件，利用 Docker 层缓存
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 拷贝源码
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# 创建数据目录（运行时挂载 volume）
RUN mkdir -p /app/data/models /app/data/datasets /app/data/logs

# 时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

EXPOSE 8080

# 环境变量默认值（可通过 docker run -e 或 compose 覆盖）
ENV HOST=0.0.0.0 \
    PORT=8080 \
    ADMIN_USERNAME=admin \
    ADMIN_PASSWORD=admin123 \
    TRAINING_MODE=real \
    MODEL_DOWNLOAD_MODE=real

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["python", "backend/main.py"]
