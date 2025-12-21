from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Response
from fastapi.staticfiles import StaticFiles

from .ui import create_ui_router

# ---------------------------------------------------------------------
# Config (single source of truth)
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Cloud-friendly default: /var/lib/gaithub
# Local dev: override with GAITHUB_DATA_DIR=/path/to/gaithub_data
DEFAULT_DATA_DIR = "/var/lib/gaithub"
DATA_DIR = Path(os.environ.get("GAITHUB_DATA_DIR", DEFAULT_DATA_DIR)).resolve()

TOKENS_JSON = os.environ.get("GAITHUB_TOKENS_JSON", "{}")
GAITHUB_VERSION = os.environ.get("GAITHUB_VERSION", "0.1.0-dev")
START_TIME = time.time()

try:
    TOKEN_MAP: Dict[str, str] = json.loads(TOKENS_JSON)  # token -> username
except Exception:
    TOKEN_MAP = {}


# Auth mode
#  - GAITHUB_REQUIRE_AUTH=1 (default): require Bearer token for writes
#  - GAITHUB_REQUIRE_AUTH=0: allow anonymous writes
REQUIRE_AUTH = os.environ.get("GAITHUB_REQUIRE_AUTH", "1").strip() != "0"

# Optional safety valve:
# If set, anonymous writes are only allowed when owner is in this comma list.
# Example: GAITHUB_PUBLIC_OWNERS=john,automateyournetwork
PUBLIC_OWNERS = {
    x.strip() for x in os.environ.get("GAITHUB_PUBLIC_OWNERS", "").split(",") if x.strip()
}

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------

app = FastAPI(title="gaithubd v0")

# Static files live under gaithubd/static
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# UI router reads from DATA_DIR
app.include_router(create_ui_router(DATA_DIR))


# ---------------------------------------------------------------------
# Meta / health
# ---------------------------------------------------------------------

def _count_repos(data_dir: Path) -> int:
    repos_root = data_dir / "repos"
    if not repos_root.exists():
        return 0

    n = 0
    for owner_dir in repos_root.iterdir():
        if not owner_dir.is_dir():
            continue
        for repo_dir in owner_dir.iterdir():
            if repo_dir.is_dir():
                n += 1
    return n


@app.get("/__meta")
def meta():
    return {
        "service": "gaithubd",
        "version": GAITHUB_VERSION,
        "data_dir": str(DATA_DIR),
        "repo_count": _count_repos(DATA_DIR),
        "uptime_seconds": int(time.time() - START_TIME),
    }


# ---------------------------------------------------------------------
# Helpers (storage + auth)
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
    # Anonymous mode
    if not REQUIRE_AUTH:
        return "anonymous"

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    user = TOKEN_MAP.get(token)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid token")
    return user


def _validate_owner(owner: str, user: str) -> None:
    # In authenticated mode: only repo owner can write (current behavior)
    if REQUIRE_AUTH:
        if owner != user:
            raise HTTPException(status_code=403, detail="Not allowed for this owner")
        return

    # Anonymous mode:
    # If PUBLIC_OWNERS is empty -> allow writing to any owner (fully public, risky)
    # If PUBLIC_OWNERS has entries -> only allow those owners
    if PUBLIC_OWNERS and owner not in PUBLIC_OWNERS:
        raise HTTPException(status_code=403, detail="Owner not enabled for anonymous writes")

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
        raise HTTPException(status_code=400, detail=f"OID mismatch: expected {oid}, got {computed}")

    path = _fanout_path(_objects_dir(owner, repo), oid)
    if path.exists():
        return {"ok": True, "stored": False}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canon + b"\n")  # store with newline for parity
    return {"ok": True, "stored": True}


@app.post("/repos/{owner}/{repo}/objects/missing")
def objects_missing(owner: str, repo: str, payload: dict):
    oids = payload.get("oids") or []
    objects_dir = _objects_dir(owner, repo)

    missing = []
    for oid in oids:
        p = _fanout_path(objects_dir, oid)
        if not p.exists():
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

    new_oid = str(payload.get("oid", "")).strip()
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

    new_oid = str(payload.get("oid", "")).strip()
    path = _refs_memory_dir(owner, repo) / branch
    old_oid = _read_ref(path)

    if if_match is not None and if_match != old_oid:
        raise HTTPException(status_code=409, detail="Ref changed")

    _write_ref(path, new_oid)
    return {"ok": True, "old": old_oid, "new": new_oid}
