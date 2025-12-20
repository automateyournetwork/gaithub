
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

from fastapi import FastAPI, Header, HTTPException, Response, Body

app = FastAPI(title="gaithubd v0")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("GAITHUB_DATA_DIR", "./data")).resolve()
TOKENS_JSON = os.environ.get("GAITHUB_TOKENS_JSON", "{}")

try:
    TOKEN_MAP: Dict[str, str] = json.loads(TOKENS_JSON)  # token -> username
except Exception:
    TOKEN_MAP = {}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _canonical_payload_bytes(raw: bytes) -> bytes:
    """
    Accept canonical JSON bytes with or without trailing newline.
    Hashing MUST exclude newline.
    """
    return raw[:-1] if raw.endswith(b"\n") else raw


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _require_user(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    user = TOKEN_MAP.get(token)

    if not user:
        raise HTTPException(status_code=403, detail="Invalid token")

    return user


def _validate_owner(owner: str, user: str) -> None:
    # v0: only repo owner can write
    if owner != user:
        raise HTTPException(status_code=403, detail="Not allowed for this owner")


def _repo_root(owner: str, repo: str) -> Path:
    return DATA_DIR / "repos" / owner / repo


def _objects_dir(owner: str, repo: str) -> Path:
    return _repo_root(owner, repo) / "objects"


def _refs_heads_dir(owner: str, repo: str) -> Path:
    return _repo_root(owner, repo) / "refs" / "heads"


def _refs_memory_dir(owner: str, repo: str) -> Path:
    return _repo_root(owner, repo) / "refs" / "memory"


def _fanout_path(base: Path, oid: str) -> Path:
    return base / oid[:2] / oid[2:4] / oid


def _ensure_repo_dirs(owner: str, repo: str) -> None:
    _objects_dir(owner, repo).mkdir(parents=True, exist_ok=True)
    _refs_heads_dir(owner, repo).mkdir(parents=True, exist_ok=True)
    _refs_memory_dir(owner, repo).mkdir(parents=True, exist_ok=True)


def _read_ref(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _write_ref(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def _list_refs(base: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not base.exists():
        return out
    for p in base.rglob("*"):
        if p.is_file():
            out[p.relative_to(base).as_posix()] = _read_ref(p)
    return out

# ---------------------------------------------------------------------
# Repo creation (v0 helper)
# ---------------------------------------------------------------------

@app.post("/repos/{owner}/{repo}")
def create_repo(owner: str, repo: str, authorization: Optional[str] = Header(None)):
    user = _require_user(authorization)
    _validate_owner(owner, user)
    _ensure_repo_dirs(owner, repo)
    return {"ok": True, "owner": owner, "repo": repo}

# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------

@app.head("/repos/{owner}/{repo}/objects/{oid}")
def head_object(owner: str, repo: str, oid: str):
    path = _fanout_path(_objects_dir(owner, repo), oid)
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(status_code=200)


@app.get("/repos/{owner}/{repo}/objects/{oid}")
def get_object(owner: str, repo: str, oid: str):
    path = _fanout_path(_objects_dir(owner, repo), oid)
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(content=path.read_bytes(), media_type="application/octet-stream")


@app.put("/repos/{owner}/{repo}/objects/{oid}")
def put_object(
    owner: str,
    repo: str,
    oid: str,
    authorization: Optional[str] = Header(None),
    body: bytes = Body(...),
):
    user = _require_user(authorization)
    _validate_owner(owner, user)
    _ensure_repo_dirs(owner, repo)

    canon = _canonical_payload_bytes(body)
    computed = _sha256_hex(canon)

    if computed != oid:
        raise HTTPException(
            status_code=400,
            detail=f"OID mismatch: expected {oid}, got {computed}",
        )

    path = _fanout_path(_objects_dir(owner, repo), oid)
    if path.exists():
        return {"ok": True, "stored": False}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canon + b"\n")  # store with newline for parity
    return {"ok": True, "stored": True}


@app.post("/repos/{owner}/{repo}/objects/missing")
def objects_missing(owner: str, repo: str, payload: Dict[str, Any]):
    oids = payload.get("oids") or []
    missing = []

    base = _objects_dir(owner, repo)
    for oid in oids:
        if not _fanout_path(base, oid).exists():
            missing.append(oid)

    return {"missing": missing}

# ---------------------------------------------------------------------
# Refs
# ---------------------------------------------------------------------

@app.get("/repos/{owner}/{repo}/refs")
def get_refs(owner: str, repo: str):
    return {
        "heads": _list_refs(_refs_heads_dir(owner, repo)),
        "memory": _list_refs(_refs_memory_dir(owner, repo)),
    }


@app.put("/repos/{owner}/{repo}/refs/heads/{branch:path}")
def put_head_ref(
    owner: str,
    repo: str,
    branch: str,
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(None),
    if_match: Optional[str] = Header(None),
):
    user = _require_user(authorization)
    _validate_owner(owner, user)
    _ensure_repo_dirs(owner, repo)

    new_oid = payload.get("oid", "").strip()
    path = _refs_heads_dir(owner, repo) / branch
    old_oid = _read_ref(path)

    if if_match is not None and if_match != old_oid:
        raise HTTPException(status_code=409, detail="Ref changed")

    _write_ref(path, new_oid)
    return {"ok": True, "old": old_oid, "new": new_oid}


@app.put("/repos/{owner}/{repo}/refs/memory/{branch:path}")
def put_memory_ref(
    owner: str,
    repo: str,
    branch: str,
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(None),
    if_match: Optional[str] = Header(None),
):
    user = _require_user(authorization)
    _validate_owner(owner, user)
    _ensure_repo_dirs(owner, repo)

    new_oid = payload.get("oid", "").strip()
    path = _refs_memory_dir(owner, repo) / branch
    old_oid = _read_ref(path)

    if if_match is not None and if_match != old_oid:
        raise HTTPException(status_code=409, detail="Ref changed")

    _write_ref(path, new_oid)
    return {"ok": True, "old": old_oid, "new": new_oid}
