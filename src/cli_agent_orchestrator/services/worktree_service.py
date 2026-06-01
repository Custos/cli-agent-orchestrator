"""Per-agent git worktree isolation for Taime activity attribution.

Taime's signature feature lets several frontier agents work the same real
codebase without stepping on each other. The physical foundation is a git
**worktree per agent**: each isolated agent runs the real CLI in its own
checkout + branch off the project HEAD, so *which file a change lives in*
provably tells you *which agent made it* — no heuristics.

Worktrees live OUTSIDE the project tree (under ``TAIME_WORKTREES_DIR``) so the
main checkout stays pristine and the watcher never sees nested checkouts. We
reuse the same thin ``git`` subprocess pattern as ``diff_service`` rather than
pulling in a git library.

On top of the branch (the durable artifact) we expose a **non-invasive
snapshot** primitive: a dangling ``commit-tree`` object capturing the full
working tree (tracked + untracked) WITHOUT touching the index or the branch.
Per-turn snapshots become the forensic timeline; the branch is never written
behind the agent's back.
"""

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from cli_agent_orchestrator.constants import TAIME_WORKTREES_DIR, WORKTREE_LINK_DIRS

logger = logging.getLogger(__name__)


@dataclass
class WorktreeInfo:
    """Result of provisioning (or resolving) an isolated agent worktree."""

    terminal_key: str
    project_root: str
    repo_root: Optional[str]
    worktree_path: str
    branch: Optional[str]
    base_sha: Optional[str]
    mode: str  # "worktree" | "shared"
    error: Optional[str] = None


def _run_git(
    cwd: str, args: List[str], timeout: float = 30.0, env: Optional[dict] = None
) -> subprocess.CompletedProcess:
    """Run ``git`` in ``cwd``. Mirrors ``diff_service._run_git`` (kept local to
    avoid a cross-service import cycle); optional ``env`` overlay for the
    throwaway-index snapshot trick."""
    run_env = None
    if env is not None:
        run_env = {**os.environ, **env}
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=run_env,
    )


def is_git_repo(path: str) -> bool:
    """True if ``path`` is inside a git work tree."""
    try:
        r = _run_git(path, ["rev-parse", "--is-inside-work-tree"])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return r.returncode == 0 and r.stdout.strip() == "true"


def get_repo_root(path: str) -> Optional[str]:
    """Absolute top level of the work tree containing ``path``, or None."""
    try:
        r = _run_git(path, ["rev-parse", "--show-toplevel"])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def get_head_sha(path: str) -> Optional[str]:
    """Resolved HEAD commit SHA, or None for an unborn branch (zero-commit repo)."""
    try:
        r = _run_git(path, ["rev-parse", "HEAD"])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _project_slug(repo_root: str) -> str:
    """Stable, filesystem-safe per-project folder name (basename + short hash)."""
    base = Path(repo_root).name or "repo"
    digest = hashlib.sha1(repo_root.encode("utf-8")).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in base)
    return f"{safe}-{digest}"


def _link_gitignored_deps(repo_root: str, worktree_path: str) -> None:
    """Symlink heavy gitignored build dirs from the main checkout into the fresh
    worktree, and copy small ``.env*`` files, so the agent doesn't start with a
    broken build. Best-effort: never fail worktree creation over this."""
    src_root = Path(repo_root)
    dst_root = Path(worktree_path)
    for name in WORKTREE_LINK_DIRS:
        src = src_root / name
        dst = dst_root / name
        try:
            if src.exists() and not dst.exists():
                os.symlink(src, dst, target_is_directory=src.is_dir())
                logger.debug("worktree: linked %s -> %s", dst, src)
        except OSError as e:
            logger.debug("worktree: could not link %s: %s", name, e)
    # Copy top-level env files (small, often required, gitignored).
    try:
        for env_file in src_root.glob(".env*"):
            if env_file.is_file():
                target = dst_root / env_file.name
                if not target.exists():
                    shutil.copy2(env_file, target)
    except OSError as e:
        logger.debug("worktree: could not copy env files: %s", e)


def ensure_worktree(project_root: str, terminal_key: str, provider: str = "agent") -> WorktreeInfo:
    """Provision (or resolve) an isolated worktree for an agent.

    Returns ``mode="worktree"`` with a ready checkout on success. If the project
    is not a git repo, or has no commits yet (unborn HEAD), or git fails, returns
    ``mode="shared"`` so the caller transparently falls back to the shared dir —
    this is the graceful degradation path, not an error.
    """
    repo_root = get_repo_root(project_root)
    if not repo_root:
        return WorktreeInfo(
            terminal_key=terminal_key,
            project_root=project_root,
            repo_root=None,
            worktree_path=project_root,
            branch=None,
            base_sha=None,
            mode="shared",
            error="not a git repository",
        )

    base_sha = get_head_sha(repo_root)
    if not base_sha:
        return WorktreeInfo(
            terminal_key=terminal_key,
            project_root=project_root,
            repo_root=repo_root,
            worktree_path=project_root,
            branch=None,
            base_sha=None,
            mode="shared",
            error="repository has no commits yet",
        )

    branch = f"taime/{provider}-{terminal_key}"
    wt_dir = TAIME_WORKTREES_DIR / _project_slug(repo_root) / terminal_key
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    wt_path = str(wt_dir)

    # Idempotent: if the worktree already exists, reuse it.
    if wt_dir.exists() and is_git_repo(wt_path):
        return WorktreeInfo(
            terminal_key=terminal_key,
            project_root=project_root,
            repo_root=repo_root,
            worktree_path=wt_path,
            branch=branch,
            base_sha=base_sha,
            mode="worktree",
        )

    add = _run_git(repo_root, ["worktree", "add", "-b", branch, wt_path, base_sha])
    if add.returncode != 0:
        # Branch name may already exist (relaunch w/ same key) — retry attaching.
        retry = _run_git(repo_root, ["worktree", "add", wt_path, branch])
        if retry.returncode != 0:
            logger.warning(
                "worktree add failed for %s: %s / %s",
                terminal_key,
                add.stderr.strip(),
                retry.stderr.strip(),
            )
            return WorktreeInfo(
                terminal_key=terminal_key,
                project_root=project_root,
                repo_root=repo_root,
                worktree_path=project_root,
                branch=None,
                base_sha=base_sha,
                mode="shared",
                error=f"worktree add failed: {add.stderr.strip()}",
            )

    _link_gitignored_deps(repo_root, wt_path)
    logger.info("worktree: provisioned %s on %s for %s", wt_path, branch, terminal_key)
    return WorktreeInfo(
        terminal_key=terminal_key,
        project_root=project_root,
        repo_root=repo_root,
        worktree_path=wt_path,
        branch=branch,
        base_sha=base_sha,
        mode="worktree",
    )


def remove_worktree(repo_root: str, worktree_path: str, branch: Optional[str] = None) -> bool:
    """Tear down a worktree and (best-effort) delete its branch."""
    ok = True
    rm = _run_git(repo_root, ["worktree", "remove", "--force", worktree_path])
    if rm.returncode != 0:
        logger.debug("worktree remove failed: %s", rm.stderr.strip())
        ok = False
        # Prune stale admin entry if the dir was deleted out from under git.
        _run_git(repo_root, ["worktree", "prune"])
    if branch:
        _run_git(repo_root, ["branch", "-D", branch])
    return ok


def workspace_info(path: str) -> dict:
    """Lightweight probe of a candidate project directory for the picker UI:
    does it exist, is it a git repo (→ agents can isolate), and what branch/HEAD."""
    info = {
        "path": path,
        "exists": os.path.isdir(path),
        "is_git": False,
        "repo_root": None,
        "branch": None,
        "head_short": None,
    }
    if not info["exists"]:
        return info
    if not is_git_repo(path):
        return info
    info["is_git"] = True
    info["repo_root"] = get_repo_root(path)
    head = get_head_sha(path)
    info["head_short"] = head[:8] if head else None
    try:
        r = _run_git(path, ["rev-parse", "--abbrev-ref", "HEAD"])
        if r.returncode == 0 and r.stdout.strip():
            info["branch"] = r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return info


def changed_files(worktree_path: str, ref_a: str, ref_b: str) -> List[str]:
    """Names of files differing between two refs/snapshots in a worktree."""
    if not ref_a or not ref_b:
        return []
    r = _run_git(worktree_path, ["diff", "--name-only", ref_a, ref_b])
    if r.returncode != 0:
        return []
    return [p for p in r.stdout.splitlines() if p.strip()]


def snapshot(worktree_path: str, message: str = "taime snapshot") -> Optional[str]:
    """Capture the full working tree (tracked + untracked) as a dangling commit
    WITHOUT touching the index or the branch, and return its SHA.

    Uses a throwaway ``GIT_INDEX_FILE`` so the agent's real index/branch are
    never mutated — the snapshot is pure forensic metadata.
    """
    if not is_git_repo(worktree_path):
        return None
    parent = get_head_sha(worktree_path)
    # git rejects a 0-byte index file, so hand it a path that does NOT yet exist
    # (inside a private temp dir) and let it initialize a fresh index there.
    tmp_dir = tempfile.mkdtemp(prefix="taime-index-")
    index_path = os.path.join(tmp_dir, "index")
    try:
        env = {"GIT_INDEX_FILE": index_path}
        add = _run_git(worktree_path, ["add", "-A"], env=env)
        if add.returncode != 0:
            logger.debug("snapshot add -A failed: %s", add.stderr.strip())
            return None
        tree = _run_git(worktree_path, ["write-tree"], env=env)
        if tree.returncode != 0 or not tree.stdout.strip():
            logger.debug("snapshot write-tree failed: %s", tree.stderr.strip())
            return None
        tree_sha = tree.stdout.strip()
        commit_args = ["commit-tree", tree_sha, "-m", message]
        if parent:
            commit_args += ["-p", parent]
        commit = _run_git(worktree_path, commit_args, env=env)
        if commit.returncode != 0 or not commit.stdout.strip():
            logger.debug("snapshot commit-tree failed: %s", commit.stderr.strip())
            return None
        return commit.stdout.strip()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
