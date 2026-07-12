"""频道路由：接入第三方聊天平台机器人

支持平台：
- dingtalk: 钉钉群自定义机器人（outgoing 回调）
- feishu:   飞书事件订阅（含 challenge 握手）
- wecom:    企业微信应用消息回调
- custom:   通用 JSON 格式

每个频道绑定一个本地模型，并维护独立的对话会话（多轮上下文）。
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from auth import get_current_user, require_admin
from database import get_db, SessionLocal
from models import User, Channel, AIModel, ChatSession, ChatMessage
from schemas import ChannelCreateRequest, ApiResponse
from services.chat_service import generate_reply

router = APIRouter(prefix="/api/channels", tags=["频道"])
logger = logging.getLogger(__name__)


def _channel_to_dict(ch: Channel, model_name: str = None) -> dict:
    return {
        "id": ch.id,
        "name": ch.name,
        "platform": ch.platform,
        "model_id": ch.model_id,
        "model_name": model_name,
        "webhook_url": ch.webhook_url,
        "secret": ch.secret,
        "session_id": ch.session_id,
        "is_active": ch.is_active,
        "created_at": ch.created_at.isoformat() if ch.created_at else None,
    }


@router.get("")
def list_channels(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """列出所有频道"""
    channels = db.query(Channel).order_by(Channel.created_at.desc()).all()
    result = []
    for ch in channels:
        m = db.query(AIModel).filter(AIModel.id == ch.model_id).first()
        result.append(_channel_to_dict(ch, m.name if m else None))
    return result


@router.post("")
def create_channel(
    payload: ChannelCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    """创建频道（管理员）"""
    model = db.query(AIModel).filter(AIModel.id == payload.model_id).first()
    if not model or model.status != "ready":
        raise HTTPException(status_code=400, detail="模型不可用")
    # 创建内部对话会话以维护多轮上下文
    session = ChatSession(user_id=user.id, model_id=payload.model_id, title=f"频道:{payload.name}")
    db.add(session)
    db.flush()
    ch = Channel(
        name=payload.name,
        platform=payload.platform,
        model_id=payload.model_id,
        webhook_url=payload.webhook_url,
        secret=payload.secret,
        session_id=session.id,
    )
    db.add(ch)
    db.commit()
    db.refresh(ch)
    return _channel_to_dict(ch, model.name)


@router.delete("/{channel_id}", response_model=ApiResponse)
def delete_channel(
    channel_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    """删除频道（管理员）"""
    ch = db.query(Channel).filter(Channel.id == channel_id).first()
    if not ch:
        raise HTTPException(status_code=404, detail="频道不存在")
    # 清理关联会话与消息
    if ch.session_id:
        db.query(ChatMessage).filter(ChatMessage.session_id == ch.session_id).delete()
        sess = db.query(ChatSession).filter(ChatSession.id == ch.session_id).first()
        if sess:
            db.delete(sess)
    db.delete(ch)
    db.commit()
    return ApiResponse(message="频道已删除")


def _extract_message(body: dict) -> str:
    """从各平台请求体中提取消息文本"""
    # 钉钉 outgoing
    if "text" in body and isinstance(body["text"], dict) and "content" in body["text"]:
        return str(body["text"]["content"]).strip()
    # 飞书事件 v2
    if "event" in body and isinstance(body["event"], dict):
        msg = body["event"].get("message", {})
        if msg and "content" in msg:
            try:
                c = json.loads(msg["content"])
                if "text" in c:
                    return str(c["text"]).strip()
            except Exception:
                pass
    # 企业微信
    if "xml" in body and isinstance(body["xml"], dict):
        content = body["xml"].get("Content")
        if content:
            return str(content).strip()
    # 通用：text / content / msg / message
    for k in ("text", "content", "msg", "message"):
        if k in body and isinstance(body[k], str):
            return body[k].strip()
    return ""


def _build_reply(response_text: str, platform: str) -> dict:
    """构造各平台的回复格式"""
    if platform == "dingtalk":
        return {"msgtype": "text", "text": {"content": response_text}}
    if platform == "feishu":
        return {"msg_type": "text", "content": {"text": response_text}}
    if platform == "wecom":
        return {"msgtype": "text", "text": {"content": response_text}}
    # 通用
    return {"reply": response_text, "text": response_text}


@router.post("/{channel_id}/webhook")
async def channel_webhook(
    channel_id: str,
    request: Request,
):
    """接收第三方平台消息并回复（无需登录认证，通过 channel_id 路由）"""
    db = SessionLocal()
    try:
        ch = db.query(Channel).filter(
            Channel.id == channel_id, Channel.is_active == True
        ).first()
        if not ch:
            raise HTTPException(status_code=404, detail="频道不存在或已停用")
        model = db.query(AIModel).filter(AIModel.id == ch.model_id).first()
        if not model or model.status != "ready" or not model.local_path:
            return _build_reply("（模型暂不可用）", ch.platform)

        try:
            body = await request.json()
        except Exception:
            body = {}

        # 飞书 URL 验证（challenge 握手）
        if isinstance(body, dict) and "challenge" in body:
            return {"challenge": body["challenge"]}

        user_msg = _extract_message(body) if isinstance(body, dict) else ""
        if not user_msg:
            return _build_reply("（未识别到消息内容）", ch.platform)

        # 取历史
        history_msgs = db.query(ChatMessage).filter(
            ChatMessage.session_id == ch.session_id
        ).order_by(ChatMessage.created_at.asc()).all()
        history = [m.content for m in history_msgs]

        # 保存用户消息
        db.add(ChatMessage(session_id=ch.session_id, role="user", content=user_msg))
        db.commit()

        # 调用模型生成回复（非流式，webhook 同步返回）
        try:
            reply = generate_reply(user_msg, history, model.local_path)
        except Exception as e:
            logger.exception("频道推理失败")
            reply = f"（生成回复失败：{type(e).__name__}: {e}）"

        # 保存助手回复
        db.add(ChatMessage(session_id=ch.session_id, role="assistant", content=reply))
        db.commit()

        return _build_reply(reply, ch.platform)
    finally:
        db.close()
