"""Working-tree diff for a terminal's directory.

Taime surfaces a Monaco diff of the changes an agent made before they land in the
main workspace. We shell out to ``git`` against the terminal's working directory
(resolved from the live tmux pane), which is the natural source of truth for "what
changed" in a real project. Non-git directories return an empty, non-error result.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from cli_agent_orchestrator.clients.database import get_worktree
from cli_agent_orchestrator.services import terminal_service

# Skip embedding file content above this size in structured diffs (still listed).
_MAX_CONTENT_BYTES = 512 * 1024


@dataclass
class TerminalDiff:
    working_directory: Optional[str]
    is_git: bool
    diff: str
    files_changed: int
    files: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class FileDiff:
    """One changed file with both sides reconstructed for a side-by-side view."""

    path: str
    status: str  # "added" | "modified" | "deleted" | "renamed"
    original: str
    modified: str
    additions: int
    deletions: int
    binary: bool = False
    old_path: Optional[str] = None  # for renames


def _run_git(
    cwd: str,
    args: List[str],
    timeout: float = 15.0,
    env: Optional[dict] = None,
    text: bool = True,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    run_env = {**os.environ, **env} if env else None
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=text,
        timeout=timeout,
        env=run_env,
        input=input_text,
    )


def get_terminal_diff(terminal_id: str) -> TerminalDiff:
    """Return the combined (staged + unstaged) working-tree diff for the
    terminal's current working directory.

    Includes untracked files via ``--intent-to-add`` semantics (we add an
    explicit untracked section) so brand-new agent-created files show up.
    """
    working_directory = terminal_service.get_working_directory(terminal_id)
    if not working_directory:
        return TerminalDiff(
            working_directory=None,
            is_git=False,
            diff="",
            files_changed=0,
            error="No working directory for this terminal.",
        )

    # Is this a git work tree?
    try:
        check = _run_git(working_directory, ["rev-parse", "--is-inside-work-tree"])
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return TerminalDiff(
            working_directory=working_directory,
            is_git=False,
            diff="",
            files_changed=0,
            error=f"git unavailable: {e}",
        )

    if check.returncode != 0 or check.stdout.strip() != "true":
        return TerminalDiff(
            working_directory=working_directory,
            is_git=False,
            diff="",
            files_changed=0,
            error="Not a git repository.",
        )

    # Tracked changes (staged + unstaged) vs HEAD.
    tracked = _run_git(working_directory, ["diff", "HEAD"])
    diff_text = tracked.stdout

    # Untracked files: show them as added so new agent files are visible.
    untracked = _run_git(
        working_directory, ["ls-files", "--others", "--exclude-standard"]
    )
    untracked_files = [f for f in untracked.stdout.splitlines() if f.strip()]
    for f in untracked_files:
        # Synthesize an "added file" diff section per untracked file.
        added = _run_git(
            working_directory,
            ["diff", "--no-index", "--", "/dev/null", f],
        )
        # --no-index exits 1 when there is a difference; that's expected.
        if added.stdout:
            diff_text += ("\n" if diff_text else "") + added.stdout

    # Count changed files via name-only (tracked) + untracked.
    names = _run_git(working_directory, ["diff", "HEAD", "--name-only"])
    tracked_files = [f for f in names.stdout.splitlines() if f.strip()]
    all_files = sorted(set(tracked_files) | set(untracked_files))

    return TerminalDiff(
        working_directory=working_directory,
        is_git=True,
        diff=diff_text,
        files_changed=len(all_files),
        files=all_files,
    )


def resolve_worktree_context(terminal_id: str) -> Tuple[Optional[str], str]:
    """Resolve the directory to diff and the base ref to diff against.

    Prefers the recorded worktree (so a STOPPED agent is still reviewable via
    its on-disk worktree + fork-point ``base_sha``); falls back to the live
    tmux working directory diffed against HEAD for non-isolated terminals.
    """
    wt = get_worktree(terminal_id)
    # "member" agents share the conductor's team worktree, so they resolve to the
    # same checkout + base as the owner.
    if wt and wt.get("mode") in ("worktree", "member") and wt.get("worktree_path"):
        if os.path.isdir(wt["worktree_path"]):
            return wt["worktree_path"], (wt.get("base_sha") or "HEAD")
    # Fallback: live pane cwd vs HEAD.
    try:
        cwd = terminal_service.get_working_directory(terminal_id)
    except Exception:
        cwd = None
    return cwd, "HEAD"


def _show_file(cwd: str, ref: str, path: str) -> str:
    """Content of ``path`` at ``ref`` (empty string if absent/binary/oversize)."""
    r = _run_git(cwd, ["show", f"{ref}:{path}"])
    if r.returncode != 0:
        return ""
    if len(r.stdout.encode("utf-8", "ignore")) > _MAX_CONTENT_BYTES:
        return ""
    return r.stdout


def _read_worktree_file(cwd: str, path: str) -> Tuple[str, bool]:
    """Working-tree content of ``path``; returns (content, is_binary)."""
    full = os.path.join(cwd, path)
    try:
        with open(full, "rb") as fh:
            raw = fh.read(_MAX_CONTENT_BYTES + 1)
    except (OSError, FileNotFoundError):
        return "", False
    if b"\x00" in raw:
        return "", True
    if len(raw) > _MAX_CONTENT_BYTES:
        return "", False
    return raw.decode("utf-8", "replace"), False


def get_file_diffs(terminal_id: str) -> List[FileDiff]:
    """Per-file structured diff (both sides) for an agent's changes vs its base.

    Powers the Monaco side-by-side review and the Phase 4 selective merge UI.
    Returns an empty list for non-git/clean trees.
    """
    cwd, base = resolve_worktree_context(terminal_id)
    if not cwd:
        return []

    check = _run_git(cwd, ["rev-parse", "--is-inside-work-tree"])
    if check.returncode != 0 or check.stdout.strip() != "true":
        return []

    results: List[FileDiff] = []

    # Tracked changes vs base, with rename detection and per-file line counts.
    name_status = _run_git(cwd, ["diff", base, "--name-status", "-M"])
    numstat = _run_git(cwd, ["diff", base, "--numstat", "-M"])
    counts = {}
    for line in numstat.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            add, dele, path = parts[0], parts[1], parts[-1]
            counts[path] = (
                0 if add == "-" else int(add),
                0 if dele == "-" else int(dele),
                add == "-" and dele == "-",  # binary marker
            )

    for line in name_status.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code = parts[0]
        if code.startswith("R") and len(parts) >= 3:
            old_path, path = parts[1], parts[2]
            status = "renamed"
        else:
            old_path, path = None, parts[1]
            status = {"A": "added", "M": "modified", "D": "deleted"}.get(code[0], "modified")

        add, dele, binary = counts.get(path, (0, 0, False))
        original = "" if status == "added" else _show_file(cwd, base, old_path or path)
        if status == "deleted":
            modified, bin2 = "", binary
        else:
            modified, bin2 = _read_worktree_file(cwd, path)
        results.append(
            FileDiff(
                path=path,
                status=status,
                original="" if binary or bin2 else original,
                modified="" if binary or bin2 else modified,
                additions=add,
                deletions=dele,
                binary=binary or bin2,
                old_path=old_path,
            )
        )

    # Untracked files (brand-new agent files) shown as added.
    untracked = _run_git(cwd, ["ls-files", "--others", "--exclude-standard"])
    for path in (f for f in untracked.stdout.splitlines() if f.strip()):
        modified, binary = _read_worktree_file(cwd, path)
        line_count = 0 if binary else (modified.count("\n") + (1 if modified else 0))
        results.append(
            FileDiff(
                path=path,
                status="added",
                original="",
                modified=modified,
                additions=line_count,
                deletions=0,
                binary=binary,
            )
        )

    results.sort(key=lambda f: f.path)
    return results


# ---------------------------------------------------------------------------
# Hunk-level structured diff + selective apply (merge / revert) — Phase 4
# ---------------------------------------------------------------------------

import tempfile  # noqa: E402  (local to the apply engine)


def _worktree_tree(cwd: str) -> Optional[str]:
    """Write the full working tree (tracked + untracked) to a throwaway index and
    return its tree SHA WITHOUT touching the real index/branch."""
    tmp_dir = tempfile.mkdtemp(prefix="taime-diff-index-")
    index_path = os.path.join(tmp_dir, "index")
    try:
        env = {"GIT_INDEX_FILE": index_path}
        if _run_git(cwd, ["add", "-A"], env=env).returncode != 0:
            return None
        tree = _run_git(cwd, ["write-tree"], env=env)
        return tree.stdout.strip() if tree.returncode == 0 and tree.stdout.strip() else None
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_unified(raw: str) -> List[dict]:
    """Parse a unified diff into per-file blocks with separated hunks.

    Each file: {path, old_path, header (str), hunks: [{index, header, text,
    additions, deletions}]}. ``text`` is the full hunk (``@@`` line + body),
    ready to be concatenated under ``header`` to form a minimal applyable patch.
    """
    files: List[dict] = []
    cur: Optional[dict] = None
    for line in raw.splitlines(keepends=True):
        if line.startswith("diff --git "):
            cur = {"header_lines": [line], "hunks": [], "a_path": None, "b_path": None}
            files.append(cur)
        elif cur is None:
            continue
        elif line.startswith("@@"):
            cur["hunks"].append({"lines": [line]})
        elif cur["hunks"]:
            cur["hunks"][-1]["lines"].append(line)
        else:
            cur["header_lines"].append(line)
            if line.startswith("--- "):
                p = line[4:].strip()
                cur["a_path"] = None if p == "/dev/null" else p[2:] if p[:2] in ("a/",) else p
            elif line.startswith("+++ "):
                p = line[4:].strip()
                cur["b_path"] = None if p == "/dev/null" else p[2:] if p[:2] in ("b/",) else p

    out: List[dict] = []
    for f in files:
        path = f["b_path"] or f["a_path"] or ""
        hunks = []
        for i, h in enumerate(f["hunks"]):
            text = "".join(h["lines"])
            additions = sum(1 for ln in h["lines"][1:] if ln.startswith("+"))
            deletions = sum(1 for ln in h["lines"][1:] if ln.startswith("-"))
            hunks.append(
                {
                    "index": i,
                    "header": h["lines"][0].rstrip("\n"),
                    "text": text,
                    "additions": additions,
                    "deletions": deletions,
                }
            )
        out.append(
            {
                "path": path,
                "old_path": f["a_path"] if f["a_path"] != path else None,
                "header": "".join(f["header_lines"]),
                "hunks": hunks,
            }
        )
    return out


def get_hunked_diff(terminal_id: str) -> dict:
    """Per-file, per-hunk diff of an agent's changes vs its base.

    Returns {"terminal_id", "base", "files": [...]}, where each file carries the
    raw header + indexed hunks. The diff is computed between the base ref and a
    snapshot tree of the live working tree, so untracked agent files are included
    with standard ``a/`` ``b/`` patch paths that ``git apply`` understands.
    """
    cwd, base = resolve_worktree_context(terminal_id)
    if not cwd:
        return {"terminal_id": terminal_id, "base": None, "files": []}
    check = _run_git(cwd, ["rev-parse", "--is-inside-work-tree"])
    if check.returncode != 0 or check.stdout.strip() != "true":
        return {"terminal_id": terminal_id, "base": None, "files": []}

    tree = _worktree_tree(cwd)
    if not tree:
        return {"terminal_id": terminal_id, "base": base, "files": []}
    raw = _run_git(cwd, ["diff", base, tree]).stdout
    return {"terminal_id": terminal_id, "base": base, "files": _parse_unified(raw)}


def _build_patch(parsed_files: List[dict], selections: dict) -> str:
    """Assemble a minimal patch from selected files/hunks.

    ``selections``: {path: hunk_indices | None}. ``None`` (or missing key when a
    path is listed) selects all hunks of that file.
    """
    chunks: List[str] = []
    for f in parsed_files:
        if f["path"] not in selections:
            continue
        wanted = selections[f["path"]]
        hunks = f["hunks"]
        if wanted is not None:
            wanted_set = set(wanted)
            hunks = [h for h in hunks if h["index"] in wanted_set]
        if not hunks and f["hunks"]:
            continue  # path selected but no matching hunks → skip
        body = f["header"] + "".join(h["text"] for h in hunks)
        if not body.endswith("\n"):
            body += "\n"
        chunks.append(body)
    return "".join(chunks)


def _resolve_target_dir(target: str) -> Optional[str]:
    """Resolve an apply target: "main" → the project's main checkout, otherwise a
    terminal id → that agent's worktree path."""
    if target == "main":
        # Any worktree row gives us the repo root (the main checkout).
        # Caller passes the source terminal; we resolve repo_root from it below.
        return None  # handled by caller (needs source context)
    wt = get_worktree(target)
    if wt and wt.get("worktree_path") and os.path.isdir(wt["worktree_path"]):
        return wt["worktree_path"]
    return None


def apply_selection(
    terminal_id: str,
    target: str,
    selections: dict,
    mode: str = "merge",
) -> dict:
    """Apply a selected subset of an agent's changes.

    - mode="merge": apply the selection FORWARD onto ``target`` ("main" = the
      project's main checkout, or another terminal id = that agent's worktree).
    - mode="revert": reverse-apply the selection in the agent's OWN worktree,
      discarding those changes from this agent.

    Returns {applied, target_dir, files, conflicts, error}.
    """
    src_cwd, _base = resolve_worktree_context(terminal_id)
    if not src_cwd:
        return {"applied": False, "error": "source working directory not found"}

    parsed = get_hunked_diff(terminal_id)["files"]
    patch = _build_patch(parsed, selections)
    if not patch.strip():
        return {"applied": False, "error": "empty selection"}

    if mode == "revert":
        target_dir = src_cwd
        args = ["apply", "--reverse", "--recount", "-"]
    else:  # merge
        if target == "main":
            wt = get_worktree(terminal_id)
            target_dir = (wt or {}).get("repo_root")
        else:
            target_dir = _resolve_target_dir(target)
        if not target_dir or not os.path.isdir(target_dir):
            return {"applied": False, "error": f"target '{target}' not resolvable"}
        # --3way uses blob ancestry for clean conflict markers when context drifts.
        args = ["apply", "--3way", "--recount", "-"]

    applied = _run_git(target_dir, args, input_text=patch)
    if applied.returncode == 0:
        files = sorted(selections.keys())
        return {
            "applied": True,
            "target_dir": target_dir,
            "files": files,
            "conflicts": [],
        }

    # --3way leaves conflict markers and returns non-zero; surface that honestly.
    stderr = applied.stderr.strip()
    conflicted = "conflict" in stderr.lower() or "with conflicts" in stderr.lower()
    return {
        "applied": False,
        "target_dir": target_dir,
        "files": sorted(selections.keys()),
        "conflicts": [stderr] if conflicted else [],
        "error": stderr or "git apply failed",
    }


def get_session_contention(session_name: str) -> List[dict]:
    """Files changed by more than one agent in a session (collision risk).

    Returns [{path, terminals: [...]}] computed from each worktree's changed-file
    set vs its own base. Pure overlap detection — no merge is attempted.
    """
    from cli_agent_orchestrator.clients.database import list_worktrees_by_session

    by_path: dict = {}
    for wt in list_worktrees_by_session(session_name):
        cwd = wt.get("worktree_path")
        base = wt.get("base_sha") or "HEAD"
        tid = wt.get("terminal_id")
        if not cwd or not os.path.isdir(cwd):
            continue
        tree = _worktree_tree(cwd)
        if not tree:
            continue
        names = _run_git(cwd, ["diff", base, tree, "--name-only"]).stdout
        for path in (p for p in names.splitlines() if p.strip()):
            by_path.setdefault(path, set()).add(tid)
    return [
        {"path": p, "terminals": sorted(t for t in terms if t)}
        for p, terms in sorted(by_path.items())
        if len(terms) > 1
    ]
