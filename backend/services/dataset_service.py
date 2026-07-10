"""数据集处理服务：支持 docx / txt / 网页 URL"""
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

    if suffix == ".docx":
        text = _parse_docx(stored)
    elif suffix in (".txt", ".md", ".csv"):
        text = _parse_txt(stored)
    else:
        text = stored.read_text(encoding="utf-8", errors="ignore")

    return _create_dataset_record(source_name, "file", filename, str(stored), text)


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
    if dataset.file_path and Path(dataset.file_path).exists():
        return Path(dataset.file_path).read_text(encoding="utf-8", errors="ignore")
    return ""
