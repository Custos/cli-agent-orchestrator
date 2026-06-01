"""Activity attribution service (Taime).

Turns the desktop watcher's raw filesystem events into attributed rows in the
activity graph. In worktree mode attribution is CERTAIN: the watched dir maps
1:1 to a terminal, so the file change provably belongs to that agent. In shared
mode the same call still records the change against the supplied (active)
terminal, but tags it ``confidence="heuristic"`` so the UI can be honest about
ambiguity.

Each fs change is also linked to the agent's currently-open turn (if any), so
the per-turn timeline in the graph view is populated for free.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cli_agent_orchestrator.clients import database

logger = logging.getLogger(__name__)


def _ts_from_millis(ms: Optional[int]) -> Optional[datetime]:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def record_fs_events(terminal_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Persist a batch of filesystem-change events attributed to ``terminal_id``.

    Returns a small summary (recorded count, attribution mode/confidence).
    """
    worktree = database.get_worktree(terminal_id)
    if worktree:
        session_name = worktree.get("session_name")
        provider = worktree.get("provider")
        mode = worktree.get("mode", "shared")
    else:
        meta = database.get_terminal_metadata(terminal_id)
        session_name = meta.get("tmux_session") if meta else None
        provider = meta.get("provider") if meta else None
        mode = "shared"

    # Attribution. In an isolated worktree shared by a single agent → certain.
    # When the worktree is a TEAM (owner + delegated members share the checkout),
    # attribute the change to whichever teammate currently has an open turn
    # (processing-state correlation); the authoritative per-agent answer is the
    # per-turn snapshot diff. Plain shared-dir → heuristic.
    attributed_terminal = terminal_id
    if worktree and mode in ("worktree", "member"):
        team = database.list_worktrees_by_path(worktree.get("worktree_path", ""))
        if len(team) <= 1:
            confidence = "certain"
        else:
            processing = [
                m["terminal_id"]
                for m in team
                if database.get_open_turn(m["terminal_id"])
            ]
            if len(processing) == 1:
                attributed_terminal = processing[0]
                confidence = "inferred"
            elif len(processing) > 1:
                confidence = "contended"
            else:
                confidence = "team"
            provider = next(
                (m["provider"] for m in team if m["terminal_id"] == attributed_terminal),
                provider,
            )
    else:
        confidence = "heuristic"

    open_turn = database.get_open_turn(attributed_terminal)
    turn_id = open_turn["id"] if open_turn else None

    recorded = 0
    for ev in events:
        path = ev.get("path")
        if not path:
            continue
        meta = {"confidence": confidence, "mode": mode}
        if attributed_terminal != terminal_id:
            # Record who reported it vs who we attributed it to (team case).
            meta["reported_by"] = terminal_id
        database.record_activity(
            kind="fs_change",
            terminal_id=attributed_terminal,
            session_name=session_name,
            provider=provider,
            path=path,
            change_kind=ev.get("kind"),
            turn_id=turn_id,
            ts=_ts_from_millis(ev.get("ts")),
            meta=meta,
        )
        recorded += 1

    return {
        "recorded": recorded,
        "terminal_id": attributed_terminal,
        "mode": mode,
        "confidence": confidence,
    }


def record_checkpoint(terminal_id: str, boundary: str) -> Dict[str, Any]:
    """Open or close an agent turn at a status boundary, snapshotting its tree.

    boundary="turn_start": snapshot the worktree and open a new turn.
    boundary="turn_end": close the open turn, snapshot again, and compute the
    files it touched (precise git diff in worktree mode; fs_change events in
    shared mode). Recorded as a graph node ("checkpoint" activity event).
    """
    from cli_agent_orchestrator.services import worktree_service

    worktree = database.get_worktree(terminal_id)
    session_name = worktree.get("session_name") if worktree else None
    provider = worktree.get("provider") if worktree else None
    wt_path = worktree.get("worktree_path") if worktree else None
    # Members share the team worktree → snapshot it for their turns too, so a
    # delegated agent gets CERTAIN per-turn file attribution.
    is_wt = bool(worktree and worktree.get("mode") in ("worktree", "member") and wt_path)

    if boundary == "turn_start":
        snap = worktree_service.snapshot(wt_path, "taime turn start") if is_wt else None
        turn = database.start_turn(terminal_id, session_name, start_snapshot=snap)
        database.record_activity(
            kind="turn_start",
            terminal_id=terminal_id,
            session_name=session_name,
            provider=provider,
            turn_id=turn["id"],
            snapshot_sha=snap,
        )
        return {"turn_id": turn["id"], "turn_index": turn["turn_index"], "snapshot": snap}

    # turn_end
    open_turn = database.get_open_turn(terminal_id)
    if not open_turn:
        # No matching start (e.g. app restarted mid-turn) — open + close one.
        open_turn = database.start_turn(terminal_id, session_name)
    end_snap = worktree_service.snapshot(wt_path, "taime turn end") if is_wt else None

    files: list = []
    if is_wt:
        start_snap = None
        for t in database.list_turns(session_name or ""):
            if t["id"] == open_turn["id"]:
                start_snap = t.get("start_snapshot")
                break
        if start_snap and end_snap:
            files = worktree_service.changed_files(wt_path, start_snap, end_snap)
    if not files:
        # Fallback: distinct fs_change paths attributed to this terminal recently.
        rows = database.list_activity(terminal_id=terminal_id, limit=200)
        files = sorted({r["path"] for r in rows if r["kind"] == "fs_change" and r["path"]})

    database.end_turn(open_turn["id"], end_snapshot=end_snap, files_touched=files)
    database.record_activity(
        kind="turn_end",
        terminal_id=terminal_id,
        session_name=session_name,
        provider=provider,
        turn_id=open_turn["id"],
        snapshot_sha=end_snap,
        meta={"files_touched": files},
    )
    return {"turn_id": open_turn["id"], "files_touched": files, "snapshot": end_snap}


def file_attribution(terminal_id: str) -> Dict[str, Any]:
    """Per-file authorship for a (team) worktree review.

    Answers "who last changed this file, in which turn" by walking the team's
    per-turn snapshots (each turn records the files it touched). The latest turn
    to touch a file owns it; files touched by >1 agent are flagged as shared.

    Returns {"team": [...members...], "files": {path: {last, contributors}}}.
    For a non-isolated terminal returns empty (the diff is the agent's own).
    """
    wt = database.get_worktree(terminal_id)
    if not wt or wt.get("mode") not in ("worktree", "member"):
        return {"team": [], "files": {}}

    path = wt.get("worktree_path") or ""
    team = database.list_worktrees_by_path(path)
    team_ids = {t["terminal_id"] for t in team}
    provider_by_id = {t["terminal_id"]: t.get("provider") for t in team}

    turns = database.list_turns(wt.get("session_name") or "")
    team_turns = [t for t in turns if t["terminal_id"] in team_ids]
    # Order so the LATEST turn to touch a file wins "last".
    team_turns.sort(key=lambda t: (t.get("ended_at") or t.get("started_at") or datetime.min))

    files: Dict[str, Any] = {}
    turn_map: Dict[Any, Any] = {}  # (terminal_id, turn_index) -> turn row (for snapshots)
    for t in team_turns:
        turn_map[(t["terminal_id"], t["turn_index"])] = t
        ended = t.get("ended_at")
        contrib = {
            "terminal_id": t["terminal_id"],
            "provider": provider_by_id.get(t["terminal_id"]),
            "turn_index": t["turn_index"],
            "ended_at": ended.isoformat() if ended else None,
        }
        for f in t.get("files_touched") or []:
            entry = files.setdefault(f, {"last": None, "contributors": []})
            if not any(
                c["terminal_id"] == contrib["terminal_id"]
                and c["turn_index"] == contrib["turn_index"]
                for c in entry["contributors"]
            ):
                entry["contributors"].append(contrib)
            entry["last"] = contrib  # ascending order → last wins

    # Per-HUNK authorship for files touched by >1 teammate: attribute each
    # current hunk to the turn whose edits best match its added lines (content
    # match, drift-resistant). Turns "review per hunk" into real guidance.
    shared = [p for p, info in files.items() if len(info["contributors"]) > 1]
    if shared:
        from cli_agent_orchestrator.services import diff_service

        cwd, base = diff_service.resolve_worktree_context(terminal_id)
        if cwd:
            hunks_by_path = {
                f["path"]: f["hunks"] for f in diff_service.get_hunked_diff(terminal_id)["files"]
            }
            for p in shared:
                cur_hunks = hunks_by_path.get(p, [])
                if not cur_hunks:
                    continue
                turn_added = []  # (contributor, set(added line contents))
                for c in files[p]["contributors"]:
                    tr = turn_map.get((c["terminal_id"], c["turn_index"]))
                    if not tr or not tr.get("end_snapshot"):
                        continue
                    added = set(
                        diff_service.added_lines_for_file(
                            cwd, tr.get("start_snapshot") or base, tr["end_snapshot"], p
                        )
                    )
                    turn_added.append((c, added))
                hunk_authors: Dict[str, Any] = {}
                for h in cur_hunks:
                    hl = {
                        ln[1:]
                        for ln in h["text"].splitlines()
                        if ln.startswith("+") and not ln.startswith("+++")
                    }
                    best, best_score = None, 0
                    for c, added in turn_added:
                        score = len(hl & added)
                        if score > best_score:
                            best, best_score = c, score
                    if best and best_score > 0:
                        hunk_authors[str(h["index"])] = {
                            "terminal_id": best["terminal_id"],
                            "provider": best["provider"],
                            "turn_index": best["turn_index"],
                        }
                if hunk_authors:
                    files[p]["hunks"] = hunk_authors

    return {
        "team": [
            {
                "terminal_id": t["terminal_id"],
                "provider": t.get("provider"),
                "mode": t.get("mode"),
                "member_of": t.get("member_of"),
            }
            for t in team
        ],
        "files": files,
    }


_EDGE_KINDS = {"handoff", "assign", "send_message"}


def build_graph(session_name: str) -> Dict[str, Any]:
    """Assemble the activity graph for a session: agent nodes (with their turns),
    orchestration edges (who delegated to whom), and contended files."""
    from cli_agent_orchestrator.services import diff_service

    worktrees = {w["terminal_id"]: w for w in database.list_worktrees_by_session(session_name)}
    turns = database.list_turns(session_name)
    events = database.list_activity(session_name=session_name, limit=5000)

    # Collect every terminal that appears anywhere in this session.
    term_ids = set(worktrees.keys())
    for t in turns:
        if t["terminal_id"]:
            term_ids.add(t["terminal_id"])
    for e in events:
        if e["terminal_id"]:
            term_ids.add(e["terminal_id"])
        if e["target_terminal_id"]:
            term_ids.add(e["target_terminal_id"])

    # Provider/profile per terminal (worktree row first, then any event).
    def provider_for(tid: str) -> Optional[str]:
        if tid in worktrees and worktrees[tid].get("provider"):
            return worktrees[tid]["provider"]
        for e in events:
            if e["terminal_id"] == tid and e.get("provider"):
                return e["provider"]
        return None

    turns_by_term: Dict[str, list] = {}
    for t in turns:
        turns_by_term.setdefault(t["terminal_id"], []).append(
            {
                "id": t["id"],
                "turn_index": t["turn_index"],
                "started_at": t["started_at"].isoformat() if t["started_at"] else None,
                "ended_at": t["ended_at"].isoformat() if t["ended_at"] else None,
                "files_touched": t["files_touched"],
                "start_snapshot": t["start_snapshot"],
                "end_snapshot": t["end_snapshot"],
            }
        )

    agents = []
    for tid in sorted(term_ids):
        wt = worktrees.get(tid, {})
        agents.append(
            {
                "terminal_id": tid,
                "provider": provider_for(tid),
                "mode": wt.get("mode"),
                "branch": wt.get("branch"),
                "member_of": wt.get("member_of"),  # conductor id for team members
                "turns": turns_by_term.get(tid, []),
            }
        )

    edges = [
        {
            "kind": e["kind"],
            "source": e["terminal_id"],
            "target": e["target_terminal_id"],
            "ts": e["ts"].isoformat() if e["ts"] else None,
        }
        for e in events
        if e["kind"] in _EDGE_KINDS and e["terminal_id"] and e["target_terminal_id"]
    ]

    contention = diff_service.get_session_contention(session_name)

    return {
        "session": session_name,
        "agents": agents,
        "edges": edges,
        "contention": contention,
    }
