"""数据集处理服务：支持 docx / txt / md / csv / json / jsonl / 网页 URL"""
import json
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

from config import DATASET_DIR
from database import SessionLocal
from models import Dataset


def _parse_docx(path: Path) -> str:
    from docx import Document  # type: ignore
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_item_text(item) -> str:
    """从单个 JSON 对象提取文本。

    支持常见字段：
    - text / content / body / passage：直接取值
    - instruction + output / input + output：拼成问答对
    - messages / conversation：按 role 拼接
    - prompt + completion / response：拼成提示-补全
    - 其他：将所有字符串/数值字段拼成 key: value
    """
    if isinstance(item, str):
        return item
    if isinstance(item, (int, float)):
        return str(item)
    if not isinstance(item, dict):
        return ""

    # 1) 纯文本字段
    for k in ("text", "content", "body", "passage", "raw"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 2) 指令-输出对
    instr = item.get("instruction") or item.get("input") or item.get("prompt") or item.get("question")
    outp = item.get("output") or item.get("response") or item.get("answer") or item.get("completion")
    if instr and outp:
        return f"问：{instr}\n答：{outp}"

    # 3) messages / conversation 列表
    msgs = item.get("messages") or item.get("conversation")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if isinstance(m, dict):
                role = m.get("role", "")
                cont = m.get("content", "")
                if cont:
                    parts.append(f"{role}：{cont}" if role else str(cont))
            elif isinstance(m, str):
                parts.append(m)
        if parts:
            return "\n".join(parts)

    # 4) 退回到所有字符串/数值字段
    parts = []
    for k, v in item.items():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, (int, float)):
            parts.append(str(v))
    return "\n".join(parts)


def _parse_json(path: Path) -> str:
    """解析 JSON 文件为纯文本。

    支持以下结构：
    - 列表：[ {...}, {...} ] 每个元素提取文本后用空行分隔
    - 单对象：{...} 直接提取
    - {"data": [...] / {"text": "..."} 等嵌套：自动解包
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # JSON 解析失败，退回纯文本
        return raw

    # 解包常见的外层容器
    if isinstance(data, dict):
        for key in ("data", "items", "list", "samples", "examples"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if isinstance(data, list):
        texts = [_extract_item_text(it) for it in data]
        texts = [t for t in texts if t]
        return "\n\n".join(texts)
    return _extract_item_text(data)


def _parse_jsonl(path: Path) -> str:
    """解析 JSONL（每行一个 JSON 对象）为纯文本。"""
    texts = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # 非 JSON 行直接当文本
            texts.append(line)
            continue
        t = _extract_item_text(obj)
        if t:
            texts.append(t)
    return "\n\n".join(texts)


def _fetch_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EaseMindBot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _count_samples(text: str) -> int:
    """按段落粗略统计样本数"""
    samples = [s.strip() for s in text.split("\n") if len(s.strip()) >= 10]
    return max(1, len(samples))


def save_file_dataset(filename: str, raw_bytes: bytes, source_name: str) -> Dataset:
    # 仅取文件名部分，防止路径穿越
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    stored = DATASET_DIR / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_name}"
    stored.write_bytes(raw_bytes)

    text = _parse_file(stored, suffix)

    return _create_dataset_record(source_name, "file", filename, str(stored), text)


def _parse_file(path: Path, suffix: str) -> str:
    """按后缀选择解析器，返回纯文本。"""
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix in (".txt", ".md", ".csv"):
        return _parse_txt(path)
    if suffix == ".json":
        return _parse_json(path)
    if suffix == ".jsonl":
        return _parse_jsonl(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def save_url_dataset(url: str, name: str | None) -> Dataset:
    text = _fetch_url(url)
    stored = DATASET_DIR / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_url.txt"
    stored.write_text(text, encoding="utf-8")
    display_name = name or urlparse(url).netloc or "网页数据集"
    return _create_dataset_record(display_name, "url", url, str(stored), text)


def _create_dataset_record(name, source_type, source_info, file_path, text) -> Dataset:
    db = SessionLocal()
    try:
        ds = Dataset(
            name=name,
            source_type=source_type,
            source_info=source_info,
            file_path=file_path,
            sample_count=_count_samples(text),
            char_count=len(text),
            content_preview=text[:500],
        )
        db.add(ds)
        db.commit()
        db.refresh(ds)
        # detach for return
        db.expunge(ds)
        return ds
    finally:
        db.close()


def get_dataset_text(dataset: Dataset) -> str:
    if not dataset.file_path:
        return ""
    p = Path(dataset.file_path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    # docx/json/jsonl 需要按结构解析后返回纯文本，否则训练时读到的是原始字节/JSON 语法
    return _parse_file(p, suffix)
