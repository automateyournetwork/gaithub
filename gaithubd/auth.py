from __future__ import annotations

import re
import json
import hashlib
import secrets
from pathlib import Path
from typing import Any, Dict
from datetime import datetime, timezone

from fastapi import HTTPException

_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$|^[a-z0-9]{1,39}$")

def _now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _sha256_hex_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def users_dir(data_dir: Path) -> Path:
    p = data_dir / "users"
    p.mkdir(parents=True, exist_ok=True)
    return p

def user_path(data_dir: Path, username: str) -> Path:
    return users_dir(data_dir) / f"{username}.json"

def validate_username(username: str) -> str:
    u = (username or "").strip().lower()
    if not u:
        raise HTTPException(status_code=400, detail="username required")
    if not _USERNAME_RE.match(u):
        raise HTTPException(status_code=400, detail="invalid username format")
    return u

def load_user(data_dir: Path, username: str) -> Dict[str, Any]:
    p = user_path(data_dir, username)
    if not p.exists():
        raise HTTPException(status_code=404, detail="user not found")
    return json.loads(p.read_text(encoding="utf-8"))

def save_user(data_dir: Path, username: str, data: Dict[str, Any]) -> None:
    p = user_path(data_dir, username)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)

def create_user_if_missing(data_dir: Path, username: str) -> Dict[str, Any]:
    u = validate_username(username)
    p = user_path(data_dir, u)
    if p.exists():
        return load_user(data_dir, u)
    user = {"username": u, "created_at": _now_iso_z(), "tokens": []}
    save_user(data_dir, u, user)
    return user

def mint_token_for_user(data_dir: Path, username: str) -> Dict[str, str]:
    u = validate_username(username)
    user = create_user_if_missing(data_dir, u)

    raw = "gaithub_" + secrets.token_urlsafe(32)
    h = _sha256_hex_text(raw)

    tokens = user.get("tokens")
    if not isinstance(tokens, list):
        tokens = []
        user["tokens"] = tokens

    next_id = f"t{len(tokens) + 1}"
    tokens.append(
        {"id": next_id, "hash": h, "created_at": _now_iso_z(), "last_used_at": None, "revoked_at": None}
    )

    save_user(data_dir, u, user)
    return {"username": u, "token": raw}
