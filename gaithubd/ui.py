from __future__ import annotations

import os
import re
import json
import shutil
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, PlainTextResponse

# Dev-only footer toggle (set to "1" in dev)
DEV_SHOW = os.environ.get("GAITHUB_UI_DEV", "0") == "1"


def create_ui_router(data_dir: Path) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    DEV_DATA_DIR = str(data_dir)

    # ----------------------------
    # Helpers (UI)
    # ----------------------------
    def ctx(request: Request, **extra):
        base = {
            "request": request,
            "dev_show": DEV_SHOW,
            "dev_data_dir": DEV_DATA_DIR,
        }
        base.update(extra)
        return base

    def repo_root(owner: str, repo: str) -> Path:
        return _safe_repo_path(owner, repo)

    SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$")

    def _safe_name(x: str) -> bool:
        return bool(x and SAFE_NAME_RE.match(x))

    def _safe_repo_path(owner: str, repo: str) -> Path:
        """
        Prevent path traversal by validating names and ensuring resolved path stays under repos root.
        """
        if not _safe_name(owner) or not _safe_name(repo):
            raise HTTPException(status_code=400, detail="invalid owner/repo name")

        root = (data_dir / "repos").resolve()
        p = (root / owner / repo).resolve()
        if root not in p.parents:
            raise HTTPException(status_code=400, detail="invalid path")
        return p

    def objects_dir(owner: str, repo: str) -> Path:
        return repo_root(owner, repo) / "objects"

    def refs_heads_dir(owner: str, repo: str) -> Path:
        return repo_root(owner, repo) / "refs" / "heads"

    def refs_memory_dir(owner: str, repo: str) -> Path:
        return repo_root(owner, repo) / "refs" / "memory"

    def fanout_path(base: Path, oid: str) -> Path:
        return base / oid[:2] / oid[2:4] / oid

    def read_ref(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def list_refs(base: Path) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not base.exists():
            return out
        for p in base.rglob("*"):
            if p.is_file():
                out[p.relative_to(base).as_posix()] = read_ref(p)
        return out

    def load_object(owner: str, repo: str, oid: str) -> Dict[str, Any]:
        p = fanout_path(objects_dir(owner, repo), oid)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Object not found: {oid}")
        raw = p.read_text(encoding="utf-8").rstrip("\n")
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw": raw}

    def short_oid(oid: str) -> str:
        return oid[:8] if oid else ""

    # --- helper: load repo meta.json (read-only for UI) ---
    def load_meta(owner: str, repo: str) -> Dict[str, Any]:
        p = repo_root(owner, repo) / "meta.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ---- GAIT schema-aware helpers ----

    def commit_created_at(obj: Dict[str, Any]) -> Optional[str]:
        if obj.get("schema") == "gait.commit.v0":
            v = obj.get("created_at")
            return v if isinstance(v, str) else None
        return None

    def commit_kind(obj: Dict[str, Any]) -> str:
        if obj.get("schema") == "gait.commit.v0":
            v = obj.get("kind")
            return v if isinstance(v, str) and v else "commit"
        return obj.get("schema") or "object"

    def commit_message(obj: Dict[str, Any]) -> str:
        if obj.get("schema") == "gait.commit.v0":
            v = obj.get("message")
            return v if isinstance(v, str) else ""
        return ""

    def commit_parents(obj: Dict[str, Any]) -> List[str]:
        if obj.get("schema") == "gait.commit.v0":
            v = obj.get("parents") or []
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
        return []

    def commit_turn_ids(obj: Dict[str, Any]) -> List[str]:
        if obj.get("schema") == "gait.commit.v0":
            v = obj.get("turn_ids") or []
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
        return []

    def list_repos() -> List[Tuple[str, str]]:
        repos_root = data_dir / "repos"
        out: List[Tuple[str, str]] = []
        if not repos_root.exists():
            return out
        for owner_dir in sorted([p for p in repos_root.iterdir() if p.is_dir()]):
            for repo_dir in sorted([p for p in owner_dir.iterdir() if p.is_dir()]):
                out.append((owner_dir.name, repo_dir.name))
        return out

    def repo_last_updated(owner: str, repo: str, head_oid: str) -> Optional[str]:
        if not head_oid:
            return None
        obj = load_object(owner, repo, head_oid)
        return commit_created_at(obj) or (obj.get("created_at") if isinstance(obj.get("created_at"), str) else None)


    def is_commit(obj: Dict[str, Any]) -> bool:
        return obj.get("schema") == "gait.commit.v0"

    def is_turn(obj: Dict[str, Any]) -> bool:
        return obj.get("schema") == "gait.turn.v0"

    def turn_user_text(turn: Dict[str, Any]) -> str:
        u = turn.get("user") or {}
        if isinstance(u, dict):
            t = u.get("text")
            return t if isinstance(t, str) else ""
        return ""

    def turn_assistant_text(turn: Dict[str, Any]) -> str:
        a = turn.get("assistant") or {}
        if isinstance(a, dict):
            t = a.get("text")
            return t if isinstance(t, str) else ""
        return ""

    def snippet(s: str, n: int = 220) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: n - 1].rstrip() + "…"

    # ----------------------------
    # Pull Requests (UI helpers)
    # ----------------------------
    def pulls_dir(owner: str, repo: str) -> Path:
        return repo_root(owner, repo) / "pulls"

    def list_pull_files(owner: str, repo: str) -> List[Path]:
        d = pulls_dir(owner, repo)
        if not d.exists():
            return []
        return sorted([p for p in d.iterdir() if p.is_file() and p.suffix == ".json"])

    def load_pull(owner: str, repo: str, pr_id: int) -> Dict[str, Any]:
        p = pulls_dir(owner, repo) / f"{pr_id:04d}.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="pull request not found")
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(status_code=500, detail="invalid pull request json")

    def save_pull(owner: str, repo: str, pr: Dict[str, Any]) -> None:
        pr_id = int(pr.get("id", 0) or 0)
        if pr_id <= 0:
            raise HTTPException(status_code=500, detail="invalid pull id")
        p = pulls_dir(owner, repo) / f"{pr_id:04d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pr, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(p)

    def list_pulls(owner: str, repo: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in list_pull_files(owner, repo):
            try:
                pr = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            # Normalize a few fields for templates
            pr_id = pr.get("id")
            if pr_id is None:
                # try to infer from filename 0001.json
                try:
                    pr_id = int(p.stem)
                except Exception:
                    pr_id = 0
            pr["id"] = int(pr_id or 0)
            pr["status"] = pr.get("status") or "unknown"
            pr["title"] = pr.get("title") or f"PR #{pr['id']}"
            pr["author"] = pr.get("author") or "unknown"
            pr["created_at"] = pr.get("created_at") or ""
            out.append(pr)

        # sort newest first by id (simple v0)
        out.sort(key=lambda x: int(x.get("id", 0)), reverse=True)
        return out

    def open_pull_count(owner: str, repo: str) -> int:
        return sum(1 for pr in list_pulls(owner, repo) if pr.get("status") == "open")

    # ----------------------------
    # UI token auth (v0, no sessions yet)
    # ----------------------------
    def _sha256_hex_text(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _find_user_for_token(raw_token: str) -> Optional[str]:
        raw_token = (raw_token or "").strip()
        if not raw_token:
            return None
        h = _sha256_hex_text(raw_token)

        users_dir = data_dir / "users"
        if not users_dir.exists():
            return None

        for p in users_dir.glob("*.json"):
            try:
                u = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            tokens = u.get("tokens") or []
            if not isinstance(tokens, list):
                continue
            for t in tokens:
                if isinstance(t, dict) and t.get("hash") == h and not t.get("revoked_at"):
                    return str(u.get("username") or "")
        return None

    def has_write_access(owner: str, repo: str, username: str) -> bool:
        if not username:
            return False
        if username == owner:
            return True
        meta = load_meta(owner, repo)
        collab = meta.get("collaborators") or {}
        write_list = collab.get("write") or []
        return username in write_list

    # ----------------------------
    # Delete 
    # ----------------------------

    def delete_branch(owner: str, repo: str, branch: str) -> None:
        ref_path = refs_heads_dir(owner, repo) / branch
        if not ref_path.exists():
            raise HTTPException(status_code=404, detail="branch not found")
        ref_path.unlink()    

    def ui_repo_with_flash(request: Request, owner: str, repo: str, msg: str = "", error: str = ""):
        heads = list_refs(refs_heads_dir(owner, repo))
        memory = list_refs(refs_memory_dir(owner, repo))
        meta = load_meta(owner, repo)
    
        default_branch = "main" if "main" in heads else (sorted(heads.keys())[0] if heads else "")
        head_oid = heads.get(default_branch, "")
        updated = repo_last_updated(owner, repo, head_oid)
    
        base_url = str(request.base_url).rstrip("/")
        clone_url = base_url
        clone_cmd = f"gait clone {base_url} --owner {owner} --repo {repo} --path ./{repo}"
    
        # convo block (keep exactly as your ui_repo logic)
        convo: List[Dict[str, Any]] = []
        commit_count = 0
        turn_count = 0
        first_turn_user = ""
        first_turn_created_at = ""
    
        if head_oid:
            cur = head_oid
            seen: set[str] = set()
            limit_commits = 50
            collected_turns: List[Dict[str, Any]] = []
    
            while cur and cur not in seen and commit_count < limit_commits:
                seen.add(cur)
                obj = load_object(owner, repo, cur)
                if not is_commit(obj):
                    break
                
                commit_count += 1
                tids = commit_turn_ids(obj)
                for tid in tids:
                    t = load_object(owner, repo, tid)
                    if not is_turn(t):
                        continue
                    turn_count += 1
                    collected_turns.append(
                        {
                            "turn_oid": tid,
                            "turn_short": short_oid(tid),
                            "created_at": t.get("created_at") if isinstance(t.get("created_at"), str) else "",
                            "user": snippet(turn_user_text(t)),
                            "assistant": snippet(turn_assistant_text(t)),
                        }
                    )
    
                parents = commit_parents(obj)
                cur = parents[0] if parents else ""
    
            if collected_turns:
                oldest = collected_turns[0]
                first_turn_user = oldest.get("user", "")
                first_turn_created_at = oldest.get("created_at", "")
    
            convo = list(reversed(collected_turns))
    
        pr_open = open_pull_count(owner, repo)
    
        return templates.TemplateResponse(
            "repo.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                heads=heads,
                memory=memory,
                meta=meta,
                default_branch=default_branch,
                updated=updated,
                short_oid=short_oid,
                clone_url=clone_url,
                clone_cmd=clone_cmd,
                convo=convo,
                commit_count=commit_count,
                turn_count=turn_count,
                first_turn_user=first_turn_user,
                first_turn_created_at=first_turn_created_at,
                pr_open=pr_open,
                branch_delete_msg=msg,
                branch_delete_error=error,
            ),
        )


    # ----------------------------
    # Routes
    # ----------------------------

    @router.get("/", response_class=HTMLResponse)
    def ui_index(request: Request):
        return templates.TemplateResponse("index.html", ctx(request))


    @router.get("/getting-started", response_class=HTMLResponse)
    def ui_getting_started(request: Request):
        base_url = str(request.base_url).rstrip("/")
        return templates.TemplateResponse(
            "getting_started.html",
            ctx(request, base_url=base_url),
        )

    @router.get("/account", response_class=HTMLResponse)
    def ui_account(request: Request):
        return templates.TemplateResponse("account.html", ctx(request))

    @router.post("/account/register", response_class=HTMLResponse)
    def ui_account_register(request: Request, username: str = Form(...)):
        base_url = str(request.base_url).rstrip("/")
        try:
            r = requests.post(f"{base_url}/auth/register", json={"username": username}, timeout=5)
            if r.status_code == 200:
                msg = f"✅ Registered: {r.json().get('username')}"
            else:
                msg = f"❌ {r.status_code}: {r.text}"
        except Exception as e:
            msg = f"❌ Error: {e}"
        return templates.TemplateResponse("account.html", ctx(request, register_msg=msg))

    @router.post("/account/token", response_class=HTMLResponse)
    def ui_account_token(request: Request, username: str = Form(...)):
        base_url = str(request.base_url).rstrip("/")
        token = None
        token_msg = None
        try:
            r = requests.post(f"{base_url}/auth/token", json={"username": username}, timeout=5)
            if r.status_code == 200:
                token = r.json().get("token")
                if not token:
                    token_msg = "❌ Token not returned"
            else:
                token_msg = f"❌ {r.status_code}: {r.text}"
        except Exception as e:
            token_msg = f"❌ Error: {e}"
        return templates.TemplateResponse("account.html", ctx(request, token=token, token_msg=token_msg))

    @router.get("/console", response_class=HTMLResponse)
    def ui_console(request: Request):
        base_url = str(request.base_url).rstrip("/")
        return templates.TemplateResponse("console.html", ctx(request, base_url=base_url))

    @router.get("/repos", response_class=HTMLResponse)
    def ui_repos(request: Request):
        return templates.TemplateResponse("repos.html", ctx(request, repos=list_repos()))

    @router.get("/repos/{owner}/{repo}", response_class=HTMLResponse)
    def ui_repo(request: Request, owner: str, repo: str):
        heads = list_refs(refs_heads_dir(owner, repo))
        memory = list_refs(refs_memory_dir(owner, repo))

        # NEW: load meta.json for display (visibility/collaborators/parent)
        meta = load_meta(owner, repo)

        # NEW: open PR count for repo.html
        pr_open = open_pull_count(owner, repo)

        default_branch = "main" if "main" in heads else (sorted(heads.keys())[0] if heads else "")
        head_oid = heads.get(default_branch, "")
        updated = repo_last_updated(owner, repo, head_oid)

        base_url = str(request.base_url).rstrip("/")
        clone_url = base_url
        clone_cmd = f"gait clone {base_url} --owner {owner} --repo {repo} --path ./{repo}"

        # ----------------------------
        # Conversation feed + Auto README (computed)
        # ----------------------------
        convo: List[Dict[str, Any]] = []
        commit_count = 0
        turn_count = 0
        first_turn_user = ""
        first_turn_created_at = ""

        if head_oid:
            cur = head_oid
            seen: set[str] = set()
            limit_commits = 50

            # We'll collect turns oldest->newest while walking back,
            # then reverse at end so newest shows first.
            collected_turns: List[Dict[str, Any]] = []

            while cur and cur not in seen and commit_count < limit_commits:
                seen.add(cur)
                obj = load_object(owner, repo, cur)

                if not is_commit(obj):
                    break

                commit_count += 1
                tids = commit_turn_ids(obj)

                for tid in tids:
                    t = load_object(owner, repo, tid)
                    if not is_turn(t):
                        continue
                    turn_count += 1
                    collected_turns.append(
                        {
                            "turn_oid": tid,
                            "turn_short": short_oid(tid),
                            "created_at": t.get("created_at") if isinstance(t.get("created_at"), str) else "",
                            "user": snippet(turn_user_text(t)),
                            "assistant": snippet(turn_assistant_text(t)),
                        }
                    )

                parents = commit_parents(obj)
                cur = parents[0] if parents else ""

            # Oldest turn is at the beginning of collected_turns
            if collected_turns:
                oldest = collected_turns[0]
                first_turn_user = oldest.get("user", "")
                first_turn_created_at = oldest.get("created_at", "")

            # For display: newest first
            convo = list(reversed(collected_turns))

            # NEW: load meta.json for display (visibility/collaborators/parent)
            meta = load_meta(owner, repo)

        return templates.TemplateResponse(
            "repo.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                heads=heads,
                memory=memory,
                meta=meta,  # NEW: pass meta to template
                default_branch=default_branch,
                updated=updated,
                short_oid=short_oid,
                # NEW: open PR count for repo.html
                pr_open=pr_open,
                clone_url=clone_url,
                clone_cmd=clone_cmd,
                # existing fields for template
                convo=convo,
                commit_count=commit_count,
                turn_count=turn_count,
                first_turn_user=first_turn_user,
                first_turn_created_at=first_turn_created_at,
            ),
        )

    @router.get("/repos/{owner}/{repo}/tree/{branch}", response_class=HTMLResponse)
    def ui_tree(request: Request, owner: str, repo: str, branch: str):
        head_ref = refs_heads_dir(owner, repo) / branch
        head_oid = read_ref(head_ref)
        if not head_oid:
            raise HTTPException(status_code=404, detail=f"Unknown branch: {branch}")

        log: List[Dict[str, Any]] = []
        cur = head_oid
        seen: set[str] = set()
        limit = 50

        while cur and cur not in seen and len(log) < limit:
            seen.add(cur)
            obj = load_object(owner, repo, cur)

            if obj.get("schema") != "gait.commit.v0":
                log.append(
                    {
                        "oid": cur,
                        "oid_short": short_oid(cur),
                        "created_at": obj.get("created_at") if isinstance(obj.get("created_at"), str) else "",
                        "kind": obj.get("schema") or "object",
                        "message": "",
                        "parents": [],
                    }
                )
                break

            parents = commit_parents(obj)
            log.append(
                {
                    "oid": cur,
                    "oid_short": short_oid(cur),
                    "created_at": commit_created_at(obj) or "",
                    "kind": commit_kind(obj),
                    "message": commit_message(obj),
                    "parents": parents,
                }
            )
            cur = parents[0] if parents else ""

        return templates.TemplateResponse(
            "tree.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                branch=branch,
                head_oid=head_oid,
                head_short=short_oid(head_oid),
                log=log,
            ),
        )

    @router.get("/repos/{owner}/{repo}/commit/{oid}", response_class=HTMLResponse)
    def ui_commit(request: Request, owner: str, repo: str, oid: str):
        obj = load_object(owner, repo, oid)
        parents = commit_parents(obj)
        turn_ids = commit_turn_ids(obj)

        return templates.TemplateResponse(
            "commit.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                oid=oid,
                oid_short=short_oid(oid),
                obj_pretty=json.dumps(obj, indent=2, ensure_ascii=False),
                parents=parents,
                turn_ids=turn_ids,
                short_oid=short_oid,
            ),
        )

    @router.get("/repos/{owner}/{repo}/turn/{oid}", response_class=HTMLResponse)
    def ui_turn(request: Request, owner: str, repo: str, oid: str):
        obj = load_object(owner, repo, oid)

        user_text = ""
        assistant_text = ""

        if obj.get("schema") == "gait.turn.v0":
            u = obj.get("user") or {}
            a = obj.get("assistant") or {}
            if isinstance(u, dict):
                user_text = str(u.get("text") or "")
            if isinstance(a, dict):
                assistant_text = str(a.get("text") or "")

        return templates.TemplateResponse(
            "turn.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                oid=oid,
                oid_short=short_oid(oid),
                user_text=user_text,
                assistant_text=assistant_text,
                obj_pretty=json.dumps(obj, indent=2, ensure_ascii=False),
            ),
        )

    @router.get("/repos/{owner}/{repo}/remote", response_class=PlainTextResponse)
    def ui_remote_file(request: Request, owner: str, repo: str):
        # This is a simple, CLI-friendly format we can parse later in gait.
        base_url = str(request.base_url).rstrip("/")
        content = (
            f"base_url={base_url}\n"
            f"owner={owner}\n"
            f"repo={repo}\n"
        )
        filename = f"{owner}-{repo}.gaithub"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return PlainTextResponse(content, headers=headers)

    @router.get("/repos/{owner}/{repo}/pulls", response_class=HTMLResponse)
    def ui_pulls(request: Request, owner: str, repo: str):
        pulls = list_pulls(owner, repo)
        return templates.TemplateResponse(
            "pulls.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                pulls=pulls,
            ),
        )

    @router.get("/repos/{owner}/{repo}/pulls/{pr_id}", response_class=HTMLResponse)
    def ui_pull(request: Request, owner: str, repo: str, pr_id: int):
        pr = load_pull(owner, repo, pr_id)
        return templates.TemplateResponse(
            "pull.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                pr=pr,
                pr_id=pr_id,
                short_oid=short_oid,
            ),
        )

    @router.post("/repos/{owner}/{repo}/branches/{branch}/delete", response_class=HTMLResponse)
    def ui_branch_delete(
        request: Request,
        owner: str,
        repo: str,
        branch: str,
        token: str = Form(""),
    ):
        # Basic auth
        username = _find_user_for_token(token)
        if not username:
            # re-render repo page with error
            return ui_repo_with_flash(request, owner, repo, error="Invalid token.")

        if not has_write_access(owner, repo, username):
            return ui_repo_with_flash(request, owner, repo, error=f"No write access for {username}.")

        # Protect default branch (and main as extra safety)
        heads = list_refs(refs_heads_dir(owner, repo))
        default_branch = "main" if "main" in heads else (sorted(heads.keys())[0] if heads else "")
        if branch == default_branch or branch == "main":
            return ui_repo_with_flash(request, owner, repo, error=f"Refusing to delete protected branch: {branch}")

        ref_path = refs_heads_dir(owner, repo) / branch
        if not ref_path.exists():
            return ui_repo_with_flash(request, owner, repo, error=f"Branch not found: {branch}")

        ref_path.unlink()
        return ui_repo_with_flash(request, owner, repo, msg=f"Deleted branch: {branch}")

    @router.post("/repos/{owner}/{repo}/delete", response_class=HTMLResponse)
    def ui_repo_delete(
        request: Request,
        owner: str,
        repo: str,
        token: str = Form(...),
        confirm: str = Form(...),
    ):
        # Load everything repo.html expects (same as ui_repo)
        def render_repo(*, delete_error: str = "", delete_success: str = ""):
            heads = list_refs(refs_heads_dir(owner, repo))
            memory = list_refs(refs_memory_dir(owner, repo))
            meta = load_meta(owner, repo)
    
            default_branch = "main" if "main" in heads else (sorted(heads.keys())[0] if heads else "")
            head_oid = heads.get(default_branch, "")
            updated = repo_last_updated(owner, repo, head_oid)
    
            base_url = str(request.base_url).rstrip("/")
            clone_url = base_url
            clone_cmd = f"gait clone {base_url} --owner {owner} --repo {repo} --path ./{repo}"
    
            # Conversation feed (same logic you already have)
            convo: List[Dict[str, Any]] = []
            commit_count = 0
            turn_count = 0
            first_turn_user = ""
            first_turn_created_at = ""
    
            if head_oid:
                cur = head_oid
                seen: set[str] = set()
                limit_commits = 50
                collected_turns: List[Dict[str, Any]] = []
    
                while cur and cur not in seen and commit_count < limit_commits:
                    seen.add(cur)
                    obj = load_object(owner, repo, cur)
                    if not is_commit(obj):
                        break
                    
                    commit_count += 1
                    tids = commit_turn_ids(obj)
    
                    for tid in tids:
                        t = load_object(owner, repo, tid)
                        if not is_turn(t):
                            continue
                        turn_count += 1
                        collected_turns.append(
                            {
                                "turn_oid": tid,
                                "turn_short": short_oid(tid),
                                "created_at": t.get("created_at") if isinstance(t.get("created_at"), str) else "",
                                "user": snippet(turn_user_text(t)),
                                "assistant": snippet(turn_assistant_text(t)),
                            }
                        )
    
                    parents = commit_parents(obj)
                    cur = parents[0] if parents else ""
    
                if collected_turns:
                    oldest = collected_turns[0]
                    first_turn_user = oldest.get("user", "")
                    first_turn_created_at = oldest.get("created_at", "")
    
                convo = list(reversed(collected_turns))
    
            pr_open = open_pull_count(owner, repo)
    
            return templates.TemplateResponse(
                "repo.html",
                ctx(
                    request,
                    owner=owner,
                    repo=repo,
                    heads=heads,
                    memory=memory,
                    meta=meta,
                    default_branch=default_branch,
                    updated=updated,
                    short_oid=short_oid,
                    clone_url=clone_url,
                    clone_cmd=clone_cmd,
                    convo=convo,
                    commit_count=commit_count,
                    turn_count=turn_count,
                    first_turn_user=first_turn_user,
                    first_turn_created_at=first_turn_created_at,
                    pr_open=pr_open,
                    delete_error=delete_error,
                    delete_success=delete_success,
                ),
            )
    
        # --- Validation that should be “friendly” (no FastAPI error pages) ---
        expected = f"{owner}/{repo}"
        if (confirm or "").strip() != expected:
            return render_repo(delete_error=f'Confirmation must equal "{expected}".')
    
        username = _find_user_for_token(token)
        if not username:
            return render_repo(delete_error="Invalid token. Paste the raw token from /auth/token.")
    
        if not has_write_access(owner, repo, username):
            return render_repo(delete_error=f"No write access for {username}.")
    
        # Strong default: only owner can delete
        if username != owner:
            return render_repo(delete_error="Only the repo owner can delete this repo.")
    
        # Safe path + delete
        try:
            repo_path = _safe_repo_path(owner, repo)  # recommended helper
            if not repo_path.exists():
                return render_repo(delete_error="Repo not found (already deleted?).")
            shutil.rmtree(repo_path)
        except Exception as e:
            return render_repo(delete_error=f"Delete failed: {e}")
    
        # After delete: go back to repos list with a friendly banner
        return templates.TemplateResponse(
            "repos.html",
            ctx(
                request,
                repos=list_repos(),
                delete_msg=f"✅ Deleted repo: {expected}",
            ),
        )


    @router.post("/repos/{owner}/{repo}/pulls/{pr_id}/merge", response_class=HTMLResponse)
    def ui_pull_merge(request: Request, owner: str, repo: str, pr_id: int, token: str = Form("")):
        # token comes from form input name="token"
        pr = load_pull(owner, repo, pr_id)

        if pr.get("status") != "open":
            return templates.TemplateResponse(
                "pull.html",
                ctx(
                    request,
                    owner=owner,
                    repo=repo,
                    pr=pr,
                    pr_id=pr_id,
                    short_oid=short_oid,
                    error="PR is not open.",
                ),
            )

        username = _find_user_for_token(token)
        if not username:
            return templates.TemplateResponse(
                "pull.html",
                ctx(
                    request,
                    owner=owner,
                    repo=repo,
                    pr=pr,
                    pr_id=pr_id,
                    short_oid=short_oid,
                    error="Invalid token (paste the raw token you received from /auth/token).",
                ),
            )

        # Require write access to target repo
        if not has_write_access(owner, repo, username):
            return templates.TemplateResponse(
                "pull.html",
                ctx(
                    request,
                    owner=owner,
                    repo=repo,
                    pr=pr,
                    pr_id=pr_id,
                    short_oid=short_oid,
                    error=f"No write access for {username}.",
                ),
            )

        # v0 merge: move target branch ref to head_oid
        head_oid = pr.get("head_oid") or ""
        to_branch = (pr.get("to") or {}).get("branch") or "main"
        if not head_oid:
            return templates.TemplateResponse(
                "pull.html",
                ctx(
                    request,
                    owner=owner,
                    repo=repo,
                    pr=pr,
                    pr_id=pr_id,
                    short_oid=short_oid,
                    error="PR has no head_oid.",
                ),
            )

        # Update ref
        ref_path = refs_heads_dir(owner, repo) / to_branch
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        old_oid = read_ref(ref_path)
        ref_path.write_text(str(head_oid).strip() + "\n", encoding="utf-8")

        # Mark PR merged
        pr["status"] = "merged"
        pr["merged_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        pr["merged_by"] = username
        pr["merged_into_old_oid"] = old_oid
        save_pull(owner, repo, pr)

        return templates.TemplateResponse(
            "pull.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                pr=pr,
                pr_id=pr_id,
                short_oid=short_oid,
                success=f"Merged by {username}. Branch {to_branch} now points to {head_oid}.",
            ),
        )

    return router
