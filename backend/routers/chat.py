"""对话路由：会话管理、发送消息(流式)、历史消息"""
import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db, SessionLocal
from models import User, ChatSession, ChatMessage, AIModel
from schemas import ChatSessionCreate, ChatSessionOut, ChatMessageOut, ChatSendRequest, ApiResponse
from services.chat_service import stream_reply

router = APIRouter(prefix="/api/chat", tags=["对话"])


@router.get("/models", response_model=list)
def chat_available_models(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """返回可对话的模型列表（ready 状态）"""
    models = db.query(AIModel).filter(AIModel.status == "ready").order_by(AIModel.created_at.desc()).all()
    # 若没有显式激活的模型，则全部可用
    return [{"id": m.id, "name": m.name, "source": m.source, "is_active": m.is_active} for m in models]


@router.post("/sessions", response_model=ChatSessionOut)
def create_session(
    payload: ChatSessionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    model = db.query(AIModel).filter(AIModel.id == payload.model_id).first()
    if not model or model.status != "ready":
        raise HTTPException(status_code=400, detail="模型不可用")
    session = ChatSession(
        user_id=user.id,
        model_id=payload.model_id,
        title=payload.title or "新对话",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return db.query(ChatSession).filter(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc()).all()


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def list_messages(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()


@router.delete("/sessions/{session_id}", response_model=ApiResponse)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.delete(session)
    db.commit()
    return ApiResponse(message="会话已删除")


@router.post("/send")
async def send_message(
    payload: ChatSendRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """流式发送消息并获取回复"""
    session = db.query(ChatSession).filter(ChatSession.id == payload.session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    model = db.query(AIModel).filter(AIModel.id == session.model_id).first()
    if not model or model.status != "ready":
        raise HTTPException(status_code=400, detail="模型不可用")

    # 在生成器外提前捕获所需值，避免生成器执行时对象已 detached
    session_id = session.id
    model_name = model.name
    model_path = model.local_path  # 真实推理的关键：本地路径

    # 先取历史（在保存当前用户消息之前，避免 autoflush 导致重复）
    history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    history_texts = [m.content for m in history]

    # 保存用户消息
    user_msg = ChatMessage(session_id=session_id, role="user", content=payload.content)
    db.add(user_msg)
    db.commit()

    user_content = payload.content

    async def event_gen():
        # 先推送用户消息确认
        yield f"data: {json.dumps({'type': 'user_message', 'content': user_content})}\n\n"
        # 推送"加载中"提示（首次加载模型可能较慢，用 hint 类型，前端可忽略）
        yield f"data: {json.dumps({'type': 'hint', 'content': '（正在加载模型，请稍候...）'})}\n\n"
        reply_buffer = ""
        try:
            for chunk in stream_reply(user_content, history_texts, model_path):
                reply_buffer += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            err_msg = f"\n\n[生成错误：{type(e).__name__}: {e}]"
            reply_buffer += err_msg
            yield f"data: {json.dumps({'type': 'token', 'content': err_msg})}\n\n"
        # 保存助手回复（使用独立 session）
        db2 = SessionLocal()
        try:
            assistant_msg = ChatMessage(session_id=session_id, role="assistant", content=reply_buffer)
            db2.add(assistant_msg)
            db2.commit()
        finally:
            db2.close()
        yield f"data: {json.dumps({'type': 'done', 'content': reply_buffer})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
