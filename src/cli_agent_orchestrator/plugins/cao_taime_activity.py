"""Taime activity-attribution plugin.

Subscribes to every CAO lifecycle/messaging event and persists it to the
``taime_activity_events`` table, turning CAO's in-memory event stream into a
durable, queryable activity graph. Orchestration events (send_message /
handoff / assign) become graph EDGES via ``target_terminal_id`` (sender ->
receiver); lifecycle events (create/kill terminal & session) become context.

This is a first-party plugin registered under the ``cao.plugins`` entry-point
group (see pyproject.toml). Hooks are best-effort and never raise into CAO —
the registry already isolates handler exceptions, and we add our own guard so a
DB hiccup can never disrupt orchestration.
"""

import logging

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.plugins.base import CaoPlugin, hook
from cli_agent_orchestrator.plugins.events import (
    PostCreateSessionEvent,
    PostCreateTerminalEvent,
    PostKillSessionEvent,
    PostKillTerminalEvent,
    PostSendMessageEvent,
)

logger = logging.getLogger(__name__)

_MESSAGE_PREVIEW_CHARS = 280


class TaimeActivityPlugin(CaoPlugin):
    """Persists CAO events into the Taime activity graph."""

    @hook("post_send_message")
    async def on_send_message(self, event: PostSendMessageEvent) -> None:
        # orchestration_type is one of send_message | handoff | assign.
        try:
            database.record_activity(
                kind=event.orchestration_type or "send_message",
                terminal_id=event.sender or None,
                target_terminal_id=event.receiver or None,
                session_name=event.session_id,
                ts=event.timestamp,
                meta={"message_preview": (event.message or "")[:_MESSAGE_PREVIEW_CHARS]},
            )
        except Exception:
            logger.warning("taime-activity: failed to record send_message", exc_info=True)

    @hook("post_create_terminal")
    async def on_create_terminal(self, event: PostCreateTerminalEvent) -> None:
        try:
            database.record_activity(
                kind="create_terminal",
                terminal_id=event.terminal_id or None,
                session_name=event.session_id,
                agent_profile=event.agent_name,
                provider=event.provider or None,
                ts=event.timestamp,
            )
        except Exception:
            logger.warning("taime-activity: failed to record create_terminal", exc_info=True)

    @hook("post_kill_terminal")
    async def on_kill_terminal(self, event: PostKillTerminalEvent) -> None:
        try:
            database.record_activity(
                kind="kill_terminal",
                terminal_id=event.terminal_id or None,
                session_name=event.session_id,
                agent_profile=event.agent_name,
                ts=event.timestamp,
            )
        except Exception:
            logger.warning("taime-activity: failed to record kill_terminal", exc_info=True)

    @hook("post_create_session")
    async def on_create_session(self, event: PostCreateSessionEvent) -> None:
        try:
            database.record_activity(
                kind="create_session",
                session_name=event.session_name or event.session_id,
                ts=event.timestamp,
            )
        except Exception:
            logger.warning("taime-activity: failed to record create_session", exc_info=True)

    @hook("post_kill_session")
    async def on_kill_session(self, event: PostKillSessionEvent) -> None:
        try:
            database.record_activity(
                kind="kill_session",
                session_name=event.session_name or event.session_id,
                ts=event.timestamp,
            )
        except Exception:
            logger.warning("taime-activity: failed to record kill_session", exc_info=True)
