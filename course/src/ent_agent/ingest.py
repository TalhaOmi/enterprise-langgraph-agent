"""Module 4 extra A: load real files into Documents (the video's PDF and directory loaders)."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .rag import Document

MAX_FILE_BYTES = 20 * 1024 * 1024  # refuse huge files: cost and denial-of-service control


def load_text_file(path: Path, tenant_id: str, allowed_roles: frozenset[str] = frozenset()) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(f"{path.stem}", tenant_id, text, path.name, allowed_roles)] if text else []


def load_pdf(path: Path, tenant_id: str, allowed_roles: frozenset[str] = frozenset()) -> list[Document]:
    """One Document per page, so a citation can name the page: id = '<file>-p<page number>'."""
    docs = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(f"{path.stem}-p{number}", tenant_id, text, f"{path.name}, page {number}", allowed_roles))
    return docs


def load_directory(folder: Path, tenant_id: str, allowed_roles: frozenset[str] = frozenset()) -> list[Document]:
    """Load every .pdf and .txt directly inside `folder` (not subfolders, not symlinks)."""
    docs: list[Document] = []
    for path in sorted(folder.iterdir()):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            continue
        if path.suffix.lower() == ".pdf":
            docs.extend(load_pdf(path, tenant_id, allowed_roles))
        elif path.suffix.lower() == ".txt":
            docs.extend(load_text_file(path, tenant_id, allowed_roles))
    return docs
