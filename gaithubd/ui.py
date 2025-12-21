from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
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
        return data_dir / "repos" / owner / repo

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
    # Routes
    # ----------------------------

    @router.get("/", response_class=HTMLResponse)
    def ui_index(request: Request):
        return templates.TemplateResponse("index.html", ctx(request))

    @router.get("/repos", response_class=HTMLResponse)
    def ui_repos(request: Request):
        return templates.TemplateResponse("repos.html", ctx(request, repos=list_repos()))

    @router.get("/repos/{owner}/{repo}", response_class=HTMLResponse)
    def ui_repo(request: Request, owner: str, repo: str):
        heads = list_refs(refs_heads_dir(owner, repo))
        memory = list_refs(refs_memory_dir(owner, repo))

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

        return templates.TemplateResponse(
            "repo.html",
            ctx(
                request,
                owner=owner,
                repo=repo,
                heads=heads,
                memory=memory,
                default_branch=default_branch,
                updated=updated,
                short_oid=short_oid,
                clone_url=clone_url,
                clone_cmd=clone_cmd,
                # new fields for template
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

    return router
