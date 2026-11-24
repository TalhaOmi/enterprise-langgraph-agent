"""Module 4 extra B: save and load the index WITHOUT pickle.

Files written to a folder:
  index.faiss      the vectors (FAISS format)
  chunks.json      the chunk texts and metadata (plain JSON)
  manifest.json    sha256 of the two files above plus an HMAC over that, signed with a secret key

Loading checks the manifest first, so a swapped or edited file is refused before it is used.
Pickle is avoided because loading a pickle file can run attacker code.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import faiss

from .rag import Chunk, Embedder, VectorStore


class IntegrityError(Exception):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign(key: bytes, index_sha: str, chunks_sha: str) -> str:
    return hmac.new(key, f"{index_sha}:{chunks_sha}".encode(), hashlib.sha256).hexdigest()


def save_store(store: VectorStore, folder: Path, key: bytes) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store._index, str(folder / "index.faiss"))
    rows = [{**c.__dict__, "allowed_roles": sorted(c.allowed_roles)} for c in store._chunks]
    (folder / "chunks.json").write_text(json.dumps(rows), encoding="utf-8")
    i, c = _sha(folder / "index.faiss"), _sha(folder / "chunks.json")
    (folder / "manifest.json").write_text(json.dumps({"index": i, "chunks": c, "hmac": _sign(key, i, c)}))


def load_store(folder: Path, embedder: Embedder, key: bytes) -> VectorStore:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    i, c = _sha(folder / "index.faiss"), _sha(folder / "chunks.json")
    if i != manifest["index"] or c != manifest["chunks"]:
        raise IntegrityError("index or chunks file changed since it was saved")
    if not hmac.compare_digest(manifest["hmac"], _sign(key, i, c)):
        raise IntegrityError("manifest signature is wrong (wrong key, or manifest was replaced)")
    store = VectorStore(embedder)
    store._index = faiss.read_index(str(folder / "index.faiss"))
    store._chunks = [
        Chunk(**{**r, "allowed_roles": frozenset(r["allowed_roles"])}) for r in json.loads((folder / "chunks.json").read_text("utf-8"))
    ]
    if store._index.ntotal != len(store._chunks) or store._index.d != embedder.dim:
        raise IntegrityError("index and chunks do not match, or embedder dimension changed")
    return store
