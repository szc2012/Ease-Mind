"""EaseMind AI 训练平台 - 主应用入口

启动后访问 http://localhost:8080
默认管理员账户：admin / admin123
"""
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import settings, BASE_DIR
from database import Base, engine, SessionLocal
from models import User
from auth import hash_password
from routers import auth as auth_router
from routers import models as models_router
from routers import datasets as datasets_router
from routers import training as training_router
from routers import distillation as distillation_router
from routers import chat as chat_router
from routers import apikeys as apikeys_router
from routers import channels as channels_router
from routers import system as system_router

FRONTEND_DIR = BASE_DIR / "frontend"


def init_db():
    """创建数据库表并初始化默认管理员"""
    Base.metadata.create_all(bind=engine)
    # 轻量迁移：为已存在的 training_tasks 表追加 evaluation 列
    _ensure_column("training_tasks", "evaluation", "JSON")
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == settings.ADMIN_USERNAME).first()
        if not admin:
            admin = User(
                username=settings.ADMIN_USERNAME,
                password_hash=hash_password(settings.ADMIN_PASSWORD),
                role="admin",
            )
            db.add(admin)
            db.commit()
            print(f"[初始化] 已创建默认管理员：{settings.ADMIN_USERNAME}（请通过环境变量修改默认密码）")
    finally:
        db.close()


def _ensure_column(table: str, column: str, ddl_type: str) -> None:
    """SQLite 轻量迁移：若表中缺少某列则追加（仅支持末尾追加）"""
    from sqlalchemy import text, inspect
    insp = inspect(engine)
    if not insp.has_table(table):
        return
    cols = {c["name"] for c in insp.get_columns(table)}
    if column in cols:
        return
    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE {table} ADD COLUMN "{column}" {ddl_type}'))
    print(f"[迁移] 已为表 {table} 追加列 {column}（{ddl_type}）")


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 注册 API 路由
app.include_router(auth_router.router)
app.include_router(models_router.router)
app.include_router(datasets_router.router)
app.include_router(training_router.router)
app.include_router(distillation_router.router)
app.include_router(chat_router.router)
app.include_router(apikeys_router.router)
app.include_router(channels_router.router)
app.include_router(system_router.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.APP_NAME}


# ---- 前端静态资源 ----
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/pages/{page_name}", response_class=HTMLResponse)
def pages(page_name: str):
    # 防止路径遍历：解析后必须仍在 pages 目录内
    pages_dir = (FRONTEND_DIR / "pages").resolve()
    p = (pages_dir / page_name).resolve()
    try:
        p.relative_to(pages_dir)
    except ValueError:
        return JSONResponse(status_code=403, content={"detail": "非法路径"})
    if not p.exists() or not p.is_file():
        return JSONResponse(status_code=404, content={"detail": "页面不存在"})
    return FileResponse(str(p))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
