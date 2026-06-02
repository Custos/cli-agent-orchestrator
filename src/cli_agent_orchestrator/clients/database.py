"""Minimal database client with only terminal metadata."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, declarative_base, sessionmaker

from cli_agent_orchestrator.constants import DATABASE_URL, DB_DIR, DEFAULT_PROVIDER
from cli_agent_orchestrator.models.flow import Flow
from cli_agent_orchestrator.models.inbox import InboxMessage, MessageStatus

logger = logging.getLogger(__name__)

Base: Any = declarative_base()


class TerminalModel(Base):
    """SQLAlchemy model for terminal metadata only."""

    __tablename__ = "terminals"

    id = Column(String, primary_key=True)  # "abc123ef"
    tmux_session = Column(String, nullable=False)  # "cao-session-name"
    tmux_window = Column(String, nullable=False)  # "window-name"
    provider = Column(String, nullable=False)  # "q_cli", "claude_code"
    agent_profile = Column(String)  # "developer", "reviewer" (optional)
    allowed_tools = Column(String, nullable=True)  # JSON-encoded list of CAO tool names
    shell_command = Column(String, nullable=True)  # shell process name captured before kiro launch
    last_active = Column(DateTime, default=datetime.now)


class InboxModel(Base):
    """SQLAlchemy model for inbox messages."""

    __tablename__ = "inbox"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sender_id = Column(String, nullable=False)
    receiver_id = Column(String, nullable=False)
    message = Column(String, nullable=False)
    status = Column(String, nullable=False)  # MessageStatus enum value
    created_at = Column(DateTime, default=datetime.now)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryMetadataModel(Base):
    """SQLAlchemy model for memory metadata (Phase 2 U1).

    SQLite is the source of truth for metadata queries; wiki markdown
    files remain the content store. Each row corresponds to exactly one
    wiki file on disk.
    """

    __tablename__ = "memory_metadata"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    memory_type = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    scope_id = Column(String, nullable=True)
    file_path = Column(String, nullable=False)
    tags = Column(String, nullable=False, default="")
    source_provider = Column(String, nullable=True)
    source_terminal_id = Column(String, nullable=True)
    token_estimate = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    __table_args__ = (UniqueConstraint("key", "scope", "scope_id", name="uq_memory_key_scope"),)


class FlowModel(Base):
    """SQLAlchemy model for flow metadata."""

    __tablename__ = "flows"

    name = Column(String, primary_key=True)
    file_path = Column(String, nullable=False)
    schedule = Column(String, nullable=False)
    agent_profile = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    script = Column(String, nullable=True)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    enabled = Column(Boolean, default=True)


class WorktreeModel(Base):
    """Per-agent git worktree provisioned for activity attribution (Taime).

    One row per terminal. ``mode``:
    - "worktree": the agent owns an isolated checkout/branch.
    - "shared": fell back to the project dir (non-git, no commits, git failure).
    - "member": a delegated sub-agent sharing the conductor's TEAM worktree
      (``member_of`` = the conductor's terminal_id). Same worktree_path/branch
      as the owner, so per-turn snapshots + team diff/merge resolve correctly.
    """

    __tablename__ = "taime_worktrees"

    terminal_id = Column(String, primary_key=True)
    session_name = Column(String, nullable=True)
    project_root = Column(String, nullable=False)
    repo_root = Column(String, nullable=True)
    worktree_path = Column(String, nullable=False)
    branch = Column(String, nullable=True)
    base_sha = Column(String, nullable=True)
    mode = Column(String, nullable=False, default="shared")
    provider = Column(String, nullable=True)
    member_of = Column(String, nullable=True)  # conductor terminal_id for "member" rows
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class AgentTurnModel(Base):
    """A single agent "turn" (one processing burst) and its tree snapshots.

    Turn boundaries are derived from the status signal (PROCESSING -> idle).
    ``start_snapshot`` / ``end_snapshot`` are dangling commit SHAs from
    ``worktree_service.snapshot`` so a per-turn diff is ``git diff start end``.
    """

    __tablename__ = "taime_agent_turns"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    terminal_id = Column(String, nullable=False)
    session_name = Column(String, nullable=True)
    turn_index = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime(timezone=True), default=_utcnow)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    start_snapshot = Column(String, nullable=True)
    end_snapshot = Column(String, nullable=True)
    files_touched = Column(String, nullable=True)  # JSON-encoded list[str]


class ActivityEventModel(Base):
    """Append-only activity-graph event.

    Unifies MCP orchestration events (send_message / handoff / assign, captured
    by the cao_taime_activity plugin) with filesystem-change events (forwarded
    by the desktop watcher) and lifecycle events into one queryable timeline.
    Edges of the graph use ``target_terminal_id`` (e.g. who handed off to whom).
    """

    __tablename__ = "taime_activity_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ts = Column(DateTime(timezone=True), default=_utcnow)
    kind = Column(String, nullable=False)  # send_message|handoff|assign|fs_change|create_terminal|...
    terminal_id = Column(String, nullable=True)
    session_name = Column(String, nullable=True)
    agent_profile = Column(String, nullable=True)
    provider = Column(String, nullable=True)
    target_terminal_id = Column(String, nullable=True)  # graph edge target
    path = Column(String, nullable=True)  # relative path for fs_change
    change_kind = Column(String, nullable=True)  # create|modify|delete
    turn_id = Column(String, nullable=True)
    snapshot_sha = Column(String, nullable=True)
    meta = Column(String, nullable=True)  # JSON-encoded extra fields


# Module-level singletons
DB_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables and apply schema migrations."""
    Base.metadata.create_all(bind=engine)
    _migrate_terminals_schema()
    _migrate_memory_indexes()
    _migrate_taime_worktrees_schema()


def _migrate_taime_worktrees_schema() -> None:
    """Add the member_of column to taime_worktrees if missing (team-worktree)."""
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(taime_worktrees)")}
            if "member_of" not in cols:
                conn.execute("ALTER TABLE taime_worktrees ADD COLUMN member_of TEXT")
                conn.commit()
                logger.info("Migration: added member_of column to taime_worktrees")
    except Exception as e:
        logger.debug(f"taime_worktrees migration skipped: {e}")


def _migrate_memory_indexes() -> None:
    """Add explicit indexes on memory_metadata for query performance."""
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_metadata (scope, scope_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_metadata (updated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_metadata (memory_type)"
            )
    except Exception as e:
        logger.debug(f"Memory index migration skipped: {e}")


def _migrate_terminals_schema() -> None:
    """Add allowed_tools and shell_command columns to terminals table if missing (schema migration)."""
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        conn = sqlite3.connect(str(DATABASE_FILE))
        cursor = conn.execute("PRAGMA table_info(terminals)")
        columns = {row[1] for row in cursor.fetchall()}
        if "allowed_tools" not in columns:
            conn.execute("ALTER TABLE terminals ADD COLUMN allowed_tools TEXT")
            conn.commit()
            logger.info("Migration: added allowed_tools column to terminals table")
        if "shell_command" not in columns:
            conn.execute("ALTER TABLE terminals ADD COLUMN shell_command TEXT")
            conn.commit()
            logger.info("Migration: added shell_command column to terminals table")
        conn.close()
    except Exception as e:
        logger.warning(f"Migration check for terminals schema failed: {e}")


def create_terminal(
    terminal_id: str,
    tmux_session: str,
    tmux_window: str,
    provider: str,
    agent_profile: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    shell_command: Optional[str] = None,
) -> Dict[str, Any]:
    """Create terminal metadata record."""
    import json as _json

    with SessionLocal() as db:
        terminal = TerminalModel(
            id=terminal_id,
            tmux_session=tmux_session,
            tmux_window=tmux_window,
            provider=provider,
            agent_profile=agent_profile,
            allowed_tools=_json.dumps(allowed_tools) if allowed_tools else None,
            shell_command=shell_command,
        )
        db.add(terminal)
        db.commit()
        return {
            "id": terminal.id,
            "tmux_session": terminal.tmux_session,
            "tmux_window": terminal.tmux_window,
            "provider": terminal.provider,
            "agent_profile": terminal.agent_profile,
            "allowed_tools": allowed_tools,
            "shell_command": terminal.shell_command,
        }


def get_terminal_metadata(terminal_id: str) -> Optional[Dict[str, Any]]:
    """Get terminal metadata by ID."""
    import json as _json

    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if not terminal:
            logger.warning(f"Terminal metadata not found for terminal_id: {terminal_id}")
            return None
        logger.debug(
            f"Retrieved terminal metadata for {terminal_id}: provider={terminal.provider}, session={terminal.tmux_session}"
        )
        allowed_tools = _json.loads(terminal.allowed_tools) if terminal.allowed_tools else None
        return {
            "id": terminal.id,
            "tmux_session": terminal.tmux_session,
            "tmux_window": terminal.tmux_window,
            "provider": terminal.provider,
            "agent_profile": terminal.agent_profile,
            "allowed_tools": allowed_tools,
            "shell_command": terminal.shell_command,
            "last_active": terminal.last_active,
        }


def list_terminals_by_session(tmux_session: str) -> List[Dict[str, Any]]:
    """List all terminals in a tmux session."""
    with SessionLocal() as db:
        terminals = db.query(TerminalModel).filter(TerminalModel.tmux_session == tmux_session).all()
        return [
            {
                "id": t.id,
                "tmux_session": t.tmux_session,
                "tmux_window": t.tmux_window,
                "provider": t.provider,
                "agent_profile": t.agent_profile,
                "last_active": t.last_active,
            }
            for t in terminals
        ]


def update_last_active(terminal_id: str) -> bool:
    """Update last active timestamp."""
    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if terminal:
            terminal.last_active = datetime.now()
            db.commit()
            return True
        return False


def update_terminal_shell_command(terminal_id: str, shell_command: str) -> bool:
    """Update the shell_command baseline for a terminal."""
    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if terminal:
            terminal.shell_command = shell_command
            db.commit()
            return True
        return False


def list_all_terminals() -> List[Dict[str, Any]]:
    """List all terminals."""
    with SessionLocal() as db:
        terminals = db.query(TerminalModel).all()
        return [
            {
                "id": t.id,
                "tmux_session": t.tmux_session,
                "tmux_window": t.tmux_window,
                "provider": t.provider,
                "agent_profile": t.agent_profile,
                "last_active": t.last_active,
            }
            for t in terminals
        ]


def list_pending_receiver_ids_by_provider(provider: str) -> List[str]:
    """List receiver terminal IDs with pending messages for a specific provider."""
    with SessionLocal() as db:
        rows = (
            db.query(InboxModel.receiver_id)
            .join(TerminalModel, TerminalModel.id == InboxModel.receiver_id)
            .filter(
                TerminalModel.provider == provider,
                InboxModel.status == MessageStatus.PENDING.value,
            )
            .distinct()
            .all()
        )
        return [row[0] for row in rows]


def delete_terminal(terminal_id: str) -> bool:
    """Delete terminal metadata."""
    with SessionLocal() as db:
        deleted = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).delete()
        db.commit()
        return deleted > 0


def delete_terminals_by_session(tmux_session: str) -> int:
    """Delete all terminals in a session."""
    with SessionLocal() as db:
        deleted = (
            db.query(TerminalModel).filter(TerminalModel.tmux_session == tmux_session).delete()
        )
        db.commit()
        return deleted


def create_inbox_message(sender_id: str, receiver_id: str, message: str) -> InboxMessage:
    """Create inbox message with status=MessageStatus.PENDING."""
    with SessionLocal() as db:
        inbox_msg = InboxModel(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message=message,
            status=MessageStatus.PENDING.value,
        )
        db.add(inbox_msg)
        db.commit()
        db.refresh(inbox_msg)
        return InboxMessage(
            id=inbox_msg.id,
            sender_id=inbox_msg.sender_id,
            receiver_id=inbox_msg.receiver_id,
            message=inbox_msg.message,
            status=MessageStatus(inbox_msg.status),
            created_at=inbox_msg.created_at,
        )


def get_pending_messages(receiver_id: str, limit: int = 1) -> List[InboxMessage]:
    """Get pending messages ordered by created_at ASC (oldest first)."""
    return get_inbox_messages(receiver_id, limit=limit, status=MessageStatus.PENDING)


def get_inbox_messages(
    receiver_id: str, limit: int = 10, status: Optional[MessageStatus] = None
) -> List[InboxMessage]:
    """Get inbox messages with optional status filter ordered by created_at ASC (oldest first).

    Args:
        receiver_id: Terminal ID to get messages for
        limit: Maximum number of messages to return (default: 10)
        status: Optional filter by message status (None = all statuses)

    Returns:
        List of inbox messages ordered by creation time (oldest first)
    """
    with SessionLocal() as db:
        query = db.query(InboxModel).filter(InboxModel.receiver_id == receiver_id)

        if status is not None:
            query = query.filter(InboxModel.status == status.value)

        messages = query.order_by(InboxModel.created_at.asc()).limit(limit).all()

        return [
            InboxMessage(
                id=msg.id,
                sender_id=msg.sender_id,
                receiver_id=msg.receiver_id,
                message=msg.message,
                status=MessageStatus(msg.status),
                created_at=msg.created_at,
            )
            for msg in messages
        ]


def update_message_status(message_id: int, status: MessageStatus) -> bool:
    """Update message status to MessageStatus.DELIVERED or MessageStatus.FAILED."""
    with SessionLocal() as db:
        message = db.query(InboxModel).filter(InboxModel.id == message_id).first()
        if message:
            message.status = status.value
            db.commit()
            return True
        return False


# Flow database functions


def create_flow(
    name: str,
    file_path: str,
    schedule: str,
    agent_profile: str,
    provider: str,
    script: str,
    next_run: datetime,
) -> Flow:
    """Create flow record."""
    with SessionLocal() as db:
        flow = FlowModel(
            name=name,
            file_path=file_path,
            schedule=schedule,
            agent_profile=agent_profile,
            provider=provider,
            script=script,
            next_run=next_run,
        )
        db.add(flow)
        db.commit()
        db.refresh(flow)
        return Flow(
            name=flow.name,
            file_path=flow.file_path,
            schedule=flow.schedule,
            agent_profile=flow.agent_profile,
            provider=flow.provider,
            script=flow.script,
            last_run=flow.last_run,
            next_run=flow.next_run,
            enabled=flow.enabled,
        )


def get_flow(name: str) -> Optional[Flow]:
    """Get flow by name."""
    with SessionLocal() as db:
        flow = db.query(FlowModel).filter(FlowModel.name == name).first()
        if not flow:
            return None
        return Flow(
            name=flow.name,
            file_path=flow.file_path,
            schedule=flow.schedule,
            agent_profile=flow.agent_profile,
            provider=flow.provider,
            script=flow.script,
            last_run=flow.last_run,
            next_run=flow.next_run,
            enabled=flow.enabled,
        )


def list_flows() -> List[Flow]:
    """List all flows."""
    with SessionLocal() as db:
        flows = db.query(FlowModel).order_by(FlowModel.next_run).all()
        return [
            Flow(
                name=f.name,
                file_path=f.file_path,
                schedule=f.schedule,
                agent_profile=f.agent_profile,
                provider=f.provider,
                script=f.script,
                last_run=f.last_run,
                next_run=f.next_run,
                enabled=f.enabled,
            )
            for f in flows
        ]


def update_flow_run_times(name: str, last_run: datetime, next_run: datetime) -> bool:
    """Update flow run times after execution."""
    with SessionLocal() as db:
        flow = db.query(FlowModel).filter(FlowModel.name == name).first()
        if flow:
            flow.last_run = last_run
            flow.next_run = next_run
            db.commit()
            return True
        return False


def update_flow_enabled(name: str, enabled: bool, next_run: Optional[datetime] = None) -> bool:
    """Update flow enabled status and optionally next_run."""
    with SessionLocal() as db:
        flow = db.query(FlowModel).filter(FlowModel.name == name).first()
        if flow:
            flow.enabled = enabled
            if next_run is not None:
                flow.next_run = next_run
            db.commit()
            return True
        return False


def delete_flow(name: str) -> bool:
    """Delete flow."""
    with SessionLocal() as db:
        deleted = db.query(FlowModel).filter(FlowModel.name == name).delete()
        db.commit()
        return deleted > 0


def get_flows_to_run() -> List[Flow]:
    """Get enabled flows where next_run <= now."""
    with SessionLocal() as db:
        now = datetime.now()
        flows = (
            db.query(FlowModel).filter(FlowModel.enabled == True, FlowModel.next_run <= now).all()
        )
        return [
            Flow(
                name=f.name,
                file_path=f.file_path,
                schedule=f.schedule,
                agent_profile=f.agent_profile,
                provider=f.provider,
                script=f.script,
                last_run=f.last_run,
                next_run=f.next_run,
                enabled=f.enabled,
            )
            for f in flows
        ]


# =============================================================================
# Taime activity-attribution functions (worktrees, turns, activity events)
# =============================================================================


def _worktree_to_dict(w: "WorktreeModel") -> Dict[str, Any]:
    return {
        "terminal_id": w.terminal_id,
        "session_name": w.session_name,
        "project_root": w.project_root,
        "repo_root": w.repo_root,
        "worktree_path": w.worktree_path,
        "branch": w.branch,
        "base_sha": w.base_sha,
        "mode": w.mode,
        "provider": w.provider,
        "member_of": w.member_of,
        "created_at": w.created_at,
    }


def upsert_worktree(
    terminal_id: str,
    project_root: str,
    worktree_path: str,
    mode: str,
    session_name: Optional[str] = None,
    repo_root: Optional[str] = None,
    branch: Optional[str] = None,
    base_sha: Optional[str] = None,
    provider: Optional[str] = None,
    member_of: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or replace the worktree record for a terminal."""
    with SessionLocal() as db:
        w = db.query(WorktreeModel).filter(WorktreeModel.terminal_id == terminal_id).first()
        if w is None:
            w = WorktreeModel(terminal_id=terminal_id)
            db.add(w)
        w.project_root = project_root
        w.worktree_path = worktree_path
        w.mode = mode
        w.session_name = session_name
        w.repo_root = repo_root
        w.branch = branch
        w.base_sha = base_sha
        w.provider = provider
        w.member_of = member_of
        db.commit()
        db.refresh(w)
        return _worktree_to_dict(w)


def get_worktree(terminal_id: str) -> Optional[Dict[str, Any]]:
    """Worktree record for a terminal, or None."""
    with SessionLocal() as db:
        w = db.query(WorktreeModel).filter(WorktreeModel.terminal_id == terminal_id).first()
        return _worktree_to_dict(w) if w else None


def get_worktree_owner_by_path(worktree_path: str) -> Optional[Dict[str, Any]]:
    """The OWNER worktree (mode="worktree") whose checkout is ``worktree_path``,
    or None. Used to detect when a delegated sub-agent is launched into an
    existing team worktree so we can enroll it as a member."""
    with SessionLocal() as db:
        w = (
            db.query(WorktreeModel)
            .filter(
                WorktreeModel.worktree_path == worktree_path,
                WorktreeModel.mode == "worktree",
            )
            .first()
        )
        return _worktree_to_dict(w) if w else None


def list_worktrees_by_path(worktree_path: str) -> List[Dict[str, Any]]:
    """All terminals sharing a worktree checkout (owner + members = a team)."""
    with SessionLocal() as db:
        rows = (
            db.query(WorktreeModel).filter(WorktreeModel.worktree_path == worktree_path).all()
        )
        return [_worktree_to_dict(w) for w in rows]


def list_worktrees_by_session(session_name: str) -> List[Dict[str, Any]]:
    """All worktree records in a session (for contention queries)."""
    with SessionLocal() as db:
        rows = db.query(WorktreeModel).filter(WorktreeModel.session_name == session_name).all()
        return [_worktree_to_dict(w) for w in rows]


def delete_worktree(terminal_id: str) -> bool:
    """Delete a worktree record."""
    with SessionLocal() as db:
        deleted = (
            db.query(WorktreeModel).filter(WorktreeModel.terminal_id == terminal_id).delete()
        )
        db.commit()
        return deleted > 0


def record_activity(
    kind: str,
    terminal_id: Optional[str] = None,
    session_name: Optional[str] = None,
    agent_profile: Optional[str] = None,
    provider: Optional[str] = None,
    target_terminal_id: Optional[str] = None,
    path: Optional[str] = None,
    change_kind: Optional[str] = None,
    turn_id: Optional[str] = None,
    snapshot_sha: Optional[str] = None,
    ts: Optional[datetime] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Append one activity-graph event; returns its id."""
    import json as _json

    with SessionLocal() as db:
        event = ActivityEventModel(
            kind=kind,
            terminal_id=terminal_id,
            session_name=session_name,
            agent_profile=agent_profile,
            provider=provider,
            target_terminal_id=target_terminal_id,
            path=path,
            change_kind=change_kind,
            turn_id=turn_id,
            snapshot_sha=snapshot_sha,
            meta=_json.dumps(meta) if meta else None,
        )
        if ts is not None:
            event.ts = ts
        db.add(event)
        db.commit()
        db.refresh(event)
        return event.id


def list_activity(
    session_name: Optional[str] = None,
    terminal_id: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Query the activity timeline (newest first) with optional filters."""
    import json as _json

    with SessionLocal() as db:
        q = db.query(ActivityEventModel)
        if session_name is not None:
            q = q.filter(ActivityEventModel.session_name == session_name)
        if terminal_id is not None:
            q = q.filter(ActivityEventModel.terminal_id == terminal_id)
        if since is not None:
            q = q.filter(ActivityEventModel.ts >= since)
        rows = q.order_by(ActivityEventModel.ts.desc()).limit(limit).all()
        return [
            {
                "id": e.id,
                "ts": e.ts,
                "kind": e.kind,
                "terminal_id": e.terminal_id,
                "session_name": e.session_name,
                "agent_profile": e.agent_profile,
                "provider": e.provider,
                "target_terminal_id": e.target_terminal_id,
                "path": e.path,
                "change_kind": e.change_kind,
                "turn_id": e.turn_id,
                "snapshot_sha": e.snapshot_sha,
                "meta": _json.loads(e.meta) if e.meta else None,
            }
            for e in rows
        ]


def start_turn(
    terminal_id: str,
    session_name: Optional[str] = None,
    start_snapshot: Optional[str] = None,
) -> Dict[str, Any]:
    """Open a new turn for a terminal, auto-incrementing turn_index."""
    with SessionLocal() as db:
        last = (
            db.query(AgentTurnModel)
            .filter(AgentTurnModel.terminal_id == terminal_id)
            .order_by(AgentTurnModel.turn_index.desc())
            .first()
        )
        next_index = (last.turn_index + 1) if last else 0
        turn = AgentTurnModel(
            terminal_id=terminal_id,
            session_name=session_name,
            turn_index=next_index,
            start_snapshot=start_snapshot,
        )
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return {"id": turn.id, "terminal_id": turn.terminal_id, "turn_index": turn.turn_index}


def get_open_turn(terminal_id: str) -> Optional[Dict[str, Any]]:
    """Most recent turn for a terminal that has not been ended yet."""
    with SessionLocal() as db:
        turn = (
            db.query(AgentTurnModel)
            .filter(AgentTurnModel.terminal_id == terminal_id, AgentTurnModel.ended_at.is_(None))
            .order_by(AgentTurnModel.turn_index.desc())
            .first()
        )
        if not turn:
            return None
        return {"id": turn.id, "terminal_id": turn.terminal_id, "turn_index": turn.turn_index}


def end_turn(
    turn_id: str,
    end_snapshot: Optional[str] = None,
    files_touched: Optional[List[str]] = None,
) -> bool:
    """Close a turn, recording its end snapshot and touched files."""
    import json as _json

    with SessionLocal() as db:
        turn = db.query(AgentTurnModel).filter(AgentTurnModel.id == turn_id).first()
        if not turn:
            return False
        turn.ended_at = _utcnow()
        turn.end_snapshot = end_snapshot
        if files_touched is not None:
            turn.files_touched = _json.dumps(files_touched)
        db.commit()
        return True


def list_turns(session_name: str) -> List[Dict[str, Any]]:
    """All turns in a session, ordered by start time."""
    import json as _json

    with SessionLocal() as db:
        rows = (
            db.query(AgentTurnModel)
            .filter(AgentTurnModel.session_name == session_name)
            .order_by(AgentTurnModel.started_at.asc())
            .all()
        )
        return [
            {
                "id": t.id,
                "terminal_id": t.terminal_id,
                "session_name": t.session_name,
                "turn_index": t.turn_index,
                "started_at": t.started_at,
                "ended_at": t.ended_at,
                "start_snapshot": t.start_snapshot,
                "end_snapshot": t.end_snapshot,
                "files_touched": _json.loads(t.files_touched) if t.files_touched else [],
            }
            for t in rows
        ]
