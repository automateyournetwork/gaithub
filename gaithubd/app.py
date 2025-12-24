from __future__ import annotations


import re
import os
import time
import json
import hashlib
import secrets
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from fastapi.staticfiles import StaticFiles
from fastapi import Body, FastAPI, Header, HTTPException, Response

from .ui import create_ui_router
from .auth import create_user_if_missing, mint_token_for_user, validate_username

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

    raw_token = authorization.split(" ", 1)[1].strip()

    # 1) Optional bootstrap override: env token map
    user = TOKEN_MAP.get(raw_token)
    if user:
        return user

    # 2) Stored user tokens (sha256 hashes)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    users_dir = DATA_DIR / "users"
    if not users_dir.exists():
        raise HTTPException(status_code=403, detail="Invalid token")

    # v0 approach: scan user files
    for p in users_dir.glob("*.json"):
        try:
            u = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        username = u.get("username")
        if not isinstance(username, str) or not username:
            continue

        tokens = u.get("tokens") or []
        if not isinstance(tokens, list):
            continue

        changed = False
        for t in tokens:
            if not isinstance(t, dict):
                continue
            if t.get("revoked_at"):
                continue
            if t.get("hash") == token_hash:
                if t.get("last_used_at") is None:
                    t["last_used_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                    changed = True
                if changed:
                    # save back to same file (atomic-ish)
                    tmp = p.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(u, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    tmp.replace(p)
                return username

    raise HTTPException(status_code=403, detail="Invalid token")

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

def _parent_repo(owner: str, repo: str) -> Optional[Dict[str, str]]:
    meta = _load_meta(owner, repo)
    parent = meta.get("parent")
    if isinstance(parent, dict):
        po = parent.get("owner")
        pr = parent.get("repo")
        if isinstance(po, str) and isinstance(pr, str) and po and pr:
            return {"owner": po, "repo": pr}
    return None

def _find_object_path(owner: str, repo: str, oid: str) -> Optional[Path]:
    # check local
    p = _fanout_path(_objects_dir(owner, repo), oid)
    if p.exists():
        return p
    # fallback to parent
    parent = _parent_repo(owner, repo)
    if parent:
        pp = _fanout_path(_objects_dir(parent["owner"], parent["repo"]), oid)
        if pp.exists():
            return pp
    return None

# ---------------------------------------------------------------------
# Repo metadata (ACL / visibility)
# ---------------------------------------------------------------------

def _meta_path(owner: str, repo: str) -> Path:
    return _repo_root(owner, repo) / "meta.json"

def _load_meta(owner: str, repo: str) -> Dict[str, Any]:
    p = _meta_path(owner, repo)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_meta_if_missing(owner: str, repo: str, creator_user: str) -> Dict[str, Any]:
    p = _meta_path(owner, repo)
    if p.exists():
        return _load_meta(owner, repo)

    meta = {
        "owner": owner,
        "repo": repo,
        "created_at": _now_iso_z(),
        "visibility": "public",  # v0 default
        "collaborators": {
            "write": [creator_user] if creator_user else [owner],
            "read": [],
        },
        "parent": None,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta

def _require_write_access(owner: str, repo: str, user: str) -> None:
    if not user:
        raise HTTPException(status_code=401, detail="Missing user")

    meta = _load_meta(owner, repo)

    # Auto-init if owner is writing the first time
    if not meta:
        if user != owner:
            raise HTTPException(status_code=403, detail="Repo not initialized for this owner")
        _ensure_repo_dirs(owner, repo)
        _write_meta_if_missing(owner, repo, user)
        return

    if user == owner:
        return

    collab = meta.get("collaborators") or {}
    write_list = collab.get("write") or []
    if isinstance(write_list, list) and user in write_list:
        return

    raise HTTPException(status_code=403, detail="No write access")


# ---------------------------------------------------------------------
# Pull requests (v0 placeholder)
# ---------------------------------------------------------------------

def _pulls_dir(owner: str, repo: str) -> Path:
    p = _repo_root(owner, repo) / "pulls"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _next_pr_id(owner: str, repo: str) -> int:
    pulls = _pulls_dir(owner, repo)
    ids = []
    for p in pulls.glob("*.json"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            pass
    return max(ids, default=0) + 1

def _copy_object_if_missing(src_owner: str, src_repo: str, dst_owner: str, dst_repo: str, oid: str) -> bool:
    src = _fanout_path(_objects_dir(src_owner, src_repo), oid)
    if not src.exists():
        return False

    dst = _fanout_path(_objects_dir(dst_owner, dst_repo), oid)
    if dst.exists():
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return True

# ---------------------------------------------------------------------
# Auth: register (v0)
# ---------------------------------------------------------------------

@app.post("/auth/register")
def auth_register(payload: Dict[str, Any] = Body(...)):
    username = validate_username(str(payload.get("username", "")))
    create_user_if_missing(DATA_DIR, username)
    return {"ok": True, "username": username}


# ---------------------------------------------------------------------
# Auth: token (v0) - issues a bearer token, stored as sha256 hash
# ---------------------------------------------------------------------

@app.post("/auth/token")
def auth_token(payload: Dict[str, Any] = Body(...)):
    username = validate_username(str(payload.get("username", "")))
    token = mint_token_for_user(DATA_DIR, username)
    return {"ok": True, **token}

# ---------------------------------------------------------------------
# Repo creation (v0 helper)
# ---------------------------------------------------------------------

@app.post("/repos/{owner}/{repo}")
def create_repo(owner: str, repo: str, authorization: Optional[str] = Header(None)):
    user = _require_user(authorization)

    # Enforce auth / public-owner rules consistently
    _validate_owner(owner, user)

    _ensure_repo_dirs(owner, repo)

    # 👇 THIS IS WHERE IT GOES
    creator = user if user != "anonymous" else owner
    _write_meta_if_missing(owner, repo, creator)

    return {"ok": True, "owner": owner, "repo": repo}

# ---------------------------------------------------------------------
# Collaborators (v0)
# ---------------------------------------------------------------------

@app.post("/repos/{owner}/{repo}/collaborators/write")
def add_write_collaborator(
    owner: str,
    repo: str,
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    actor = _require_user(authorization)

    # Only repo owner can manage collaborators (v0)
    if actor != owner:
        raise HTTPException(status_code=403, detail="Only owner can manage collaborators")

    target = _validate_username(str(payload.get("username", "")))

    # Ensure repo + meta exist
    _ensure_repo_dirs(owner, repo)
    meta = _write_meta_if_missing(owner, repo, owner)

    collab = meta.get("collaborators") or {}
    write_list = collab.get("write") or []
    if not isinstance(write_list, list):
        write_list = []

    if target not in write_list:
        write_list.append(target)

    collab["write"] = write_list
    meta["collaborators"] = collab

    _meta_path(owner, repo).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {"ok": True, "owner": owner, "repo": repo, "write": write_list}

# ---------------------------------------------------------------------
# Pull requests (v0 placeholder)
# ---------------------------------------------------------------------
@app.post("/repos/{owner}/{repo}/pulls")
def create_pull_request(
    owner: str,
    repo: str,
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    actor = _require_user(authorization)

    title = str(payload.get("title", "")).strip() or "Pull request"
    from_owner = _validate_username(payload.get("from_owner"))
    from_repo = str(payload.get("from_repo")).strip()
    from_branch = str(payload.get("from_branch", "main")).strip()
    to_branch = str(payload.get("to_branch", "main")).strip()

    # Ensure author controls the source repo
    if actor != from_owner:
        raise HTTPException(status_code=403, detail="Not allowed to create PR from this repo")

    # Resolve head and base
    head_ref = _refs_heads_dir(from_owner, from_repo) / from_branch
    base_ref = _refs_heads_dir(owner, repo) / to_branch

    head_oid = _read_ref(head_ref)
    base_oid = _read_ref(base_ref)

    if not head_oid:
        raise HTTPException(status_code=400, detail="Source branch has no head")
    if not base_oid:
        raise HTTPException(status_code=400, detail="Target branch has no head")

    pr_id = _next_pr_id(owner, repo)
    pr = {
        "id": pr_id,
        "created_at": _now_iso_z(),
        "status": "open",
        "title": title,
        "author": actor,
        "from": {"owner": from_owner, "repo": from_repo, "branch": from_branch},
        "to": {"owner": owner, "repo": repo, "branch": to_branch},
        "base_oid": base_oid,
        "head_oid": head_oid,
    }

    path = _pulls_dir(owner, repo) / f"{pr_id:04d}.json"
    path.write_text(json.dumps(pr, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"ok": True, "id": pr_id, "title": title}


@app.post("/repos/{owner}/{repo}/pulls/{id}/merge")
def merge_pull_request(
    owner: str,
    repo: str,
    id: int,
    authorization: Optional[str] = Header(None),
):
    actor = _require_user(authorization)
    _require_write_access(owner, repo, actor)

    path = _pulls_dir(owner, repo) / f"{id:04d}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PR not found")

    pr = json.loads(path.read_text(encoding="utf-8"))

    if pr.get("status") != "open":
        raise HTTPException(status_code=400, detail="PR is not open")

    to_branch = pr["to"]["branch"]
    base_ref = _refs_heads_dir(owner, repo) / to_branch
    current_base = _read_ref(base_ref)

    if current_base != pr["base_oid"]:
        raise HTTPException(status_code=409, detail="Branch moved; cannot fast-forward")

    # Ensure destination has the head object bytes (minimum viable)
    src = pr["from"]
    ok = _copy_object_if_missing(src["owner"], src["repo"], owner, repo, pr["head_oid"])
    if not ok:
        raise HTTPException(status_code=400, detail="Missing head object in source repo")

    _write_ref(base_ref, pr["head_oid"])

    pr["status"] = "merged"
    pr["merged_at"] = _now_iso_z()
    path.write_text(json.dumps(pr, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"ok": True, "merged": True, "branch": to_branch}

# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------

@app.head("/repos/{owner}/{repo}/objects/{oid}")
def head_object(owner: str, repo: str, oid: str):
    path = _find_object_path(owner, repo, oid)
    if not path:
        raise HTTPException(status_code=404)
    return Response(status_code=200)


@app.get("/repos/{owner}/{repo}/objects/{oid}")
def get_object(owner: str, repo: str, oid: str):
    path = _find_object_path(owner, repo, oid)
    if not path:
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
    _require_write_access(owner, repo, user)
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
    if not isinstance(oids, list):
        oids = []

    parent = _parent_repo(owner, repo)

    missing = []
    for oid in oids:
        if not isinstance(oid, str):
            continue

        # local
        p = _fanout_path(_objects_dir(owner, repo), oid)
        if p.exists():
            continue

        # parent
        if parent:
            pp = _fanout_path(_objects_dir(parent["owner"], parent["repo"]), oid)
            if pp.exists():
                continue

        missing.append(oid)

    return {"missing": missing}


@app.post("/repos/{owner}/{repo}/fork")
def fork_repo(
    owner: str,
    repo: str,
    payload: Dict[str, Any] = Body(default={}),
    authorization: Optional[str] = Header(None),
):
    actor = _require_user(authorization)

    into_owner = _validate_username(str(payload.get("into_owner", actor)))
    into_repo  = str(payload.get("into_repo", repo)).strip() or repo

    # v0 rule: you can only fork into your own namespace
    if into_owner != actor:
        raise HTTPException(status_code=403, detail="Can only fork into your own account")

    # Ensure source exists (meta or refs/objects)
    src_meta = _write_meta_if_missing(owner, repo, owner)

    # Create destination repo dirs
    _ensure_repo_dirs(into_owner, into_repo)

    # Copy heads refs snapshot
    src_heads = _list_refs(_refs_heads_dir(owner, repo))
    for name, oid in src_heads.items():
        _write_ref(_refs_heads_dir(into_owner, into_repo) / name, oid)

    # Write dest meta with parent pointer
    dest_meta = {
        "owner": into_owner,
        "repo": into_repo,
        "created_at": _now_iso_z(),
        "visibility": "public",
        "collaborators": {"write": [into_owner], "read": []},
        "parent": {"owner": owner, "repo": repo},
    }
    _meta_path(into_owner, into_repo).write_text(
        json.dumps(dest_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {"ok": True, "fork": f"{into_owner}/{into_repo}", "parent": f"{owner}/{repo}"}

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
    _require_write_access(owner, repo, user)
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
    _require_write_access(owner, repo, user)
    _ensure_repo_dirs(owner, repo)

    new_oid = str(payload.get("oid", "")).strip()
    path = _refs_memory_dir(owner, repo) / branch
    old_oid = _read_ref(path)

    if if_match is not None and if_match != old_oid:
        raise HTTPException(status_code=409, detail="Ref changed")

    _write_ref(path, new_oid)
    return {"ok": True, "old": old_oid, "new": new_oid}
