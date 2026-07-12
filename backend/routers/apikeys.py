"""API Key 路由：密钥管理 + OpenAI 兼容对话接口

管理接口（需登录）：
- POST   /api/apikeys            创建密钥
- GET    /api/apikeys            列出密钥
- DELETE /api/apikeys/{id}       删除密钥

对外接口（用 API Key 认证）：
- POST   /api/v1/chat/completions  OpenAI 兼容对话
"""
import secrets
import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db, SessionLocal
from models import User, ApiKey, AIModel
from schemas import ApiKeyCreateRequest, ApiKeyOut, ApiResponse, OpenAIChatRequest
from services.chat_service import stream_reply


router = APIRouter(prefix="/api", tags=["API Key"])


def _generate_key() -> str:
    """生成 API Key：em-<48位hex>"""
    return "em-" + secrets.token_hex(24)


def _key_prefix(key: str) -> str:
    """返回密钥前缀（em-xxxx...xxxx）"""
    if len(key) <= 12:
        return key
    return key[:8] + "..." + key[-4:]


# ============== 管理接口 ==============

@router.post("/apikeys", response_model=ApiKeyOut)
def create_apikey(
    payload: ApiKeyCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """创建 API Key"""
    # 如果指定了模型，校验模型可用
    if payload.model_id:
        model = db.query(AIModel).filter(AIModel.id == payload.model_id).first()
        if not model or model.status != "ready":
            raise HTTPException(status_code=400, detail="指定的模型不可用")
    key = _generate_key()
    apikey = ApiKey(
        user_id=user.id,
        key=key,
        name=payload.name,
        model_id=payload.model_id,
    )
    db.add(apikey)
    db.commit()
    db.refresh(apikey)
    result = ApiKeyOut.model_validate(apikey)
    result.key = key  # 返回完整 key（仅此次）
    result.key_prefix = _key_prefix(key)
    return result


@router.get("/apikeys", response_model=list[ApiKeyOut])
def list_apikeys(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """列出当前用户的 API Key"""
    keys = db.query(ApiKey).filter(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc()).all()
    result = []
    for k in keys:
        item = ApiKeyOut.model_validate(k)
        item.key = ""  # 列表不返回完整 key
        item.key_prefix = _key_prefix(k.key)
        result.append(item)
    return result


@router.delete("/apikeys/{key_id}", response_model=ApiResponse)
def delete_apikey(
    key_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """删除 API Key"""
    apikey = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == user.id).first()
    if not apikey:
        raise HTTPException(status_code=404, detail="密钥不存在")
    db.delete(apikey)
    db.commit()
    return ApiResponse(message="密钥已删除")


# ============== 对外接口：OpenAI 兼容 ==============

def _authenticate_api_key(authorization: str, db: Session) -> tuple:
    """用 API Key 认证，返回 (user_id, apikey) 或 raise"""
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")
    # 支持 "Bearer em-xxx" 和 "em-xxx"
    token = authorization
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token.startswith("em-"):
        raise HTTPException(status_code=401, detail="无效的 API Key 格式")
    apikey = db.query(ApiKey).filter(ApiKey.key == token).first()
    if not apikey:
        raise HTTPException(status_code=401, detail="API Key 不存在")
    if not apikey.is_active:
        raise HTTPException(status_code=403, detail="API Key 已禁用")
    # 更新使用记录
    apikey.last_used_at = datetime.utcnow()
    apikey.total_requests = (apikey.total_requests or 0) + 1
    db.commit()
    return apikey.user_id, apikey


def _resolve_model(apikey: ApiKey, model_name: str, db: Session) -> AIModel:
    """解析要使用的模型"""
    # 优先用密钥绑定的模型
    if apikey.model_id:
        model = db.query(AIModel).filter(AIModel.id == apikey.model_id).first()
        if model and model.status == "ready":
            return model
    # 按 name 匹配
    if model_name and model_name != "default":
        model = db.query(AIModel).filter(AIModel.name == model_name, AIModel.status == "ready").first()
        if model:
            return model
    # 回退：任意 ready 模型
    model = db.query(AIModel).filter(AIModel.status == "ready").order_by(AIModel.is_active.desc()).first()
    if not model:
        raise HTTPException(status_code=503, detail="没有可用的模型")
    return model


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    payload: OpenAIChatRequest,
    request: Request,
    authorization: str = Header(None),
    db: Session = Depends(get_db),
):
    """OpenAI 兼容的对话接口

    使用方式：
    ```
    curl -X POST http://localhost:8080/api/v1/chat/completions \
      -H "Authorization: Bearer em-xxxxxx" \
      -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
    ```
    """
    user_id, apikey = _authenticate_api_key(authorization, db)
    model = _resolve_model(apikey, payload.model, db)

    # 提取历史消息
    history = []
    user_content = ""
    system_content = ""
    for i, msg in enumerate(payload.messages):
        if msg.role == "system":
            system_content = msg.content
        elif msg.role == "user":
            if i == len(payload.messages) - 1:
                user_content = msg.content
            else:
                history.append(msg.content)
        elif msg.role == "assistant":
            history.append(msg.content)

    # 如果有 system 消息，拼到用户消息前面
    if system_content:
        user_content = f"{system_content}\n\n{user_content}" if user_content else system_content

    model_path = model.local_path
    model_name = model.name

    if payload.stream:
        # 流式返回（SSE，OpenAI 格式）
        async def event_gen():
            created = int(datetime.utcnow().timestamp())
            for chunk in stream_reply(user_content, history, model_path):
                data = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            # 结束标记
            done_data = {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    else:
        # 非流式返回
        created = int(datetime.utcnow().timestamp())
        reply = ""
        for chunk in stream_reply(user_content, history, model_path):
            reply += chunk
        return {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


@router.get("/v1/models")
def openai_list_models(
    authorization: str = Header(None),
    db: Session = Depends(get_db),
):
    """OpenAI 兼容的模型列表接口"""
    _authenticate_api_key(authorization, db)
    models = db.query(AIModel).filter(AIModel.status == "ready").all()
    return {
        "object": "list",
        "data": [
            {"id": m.name, "object": "model", "owned_by": "easemind"}
            for m in models
        ],
    }
