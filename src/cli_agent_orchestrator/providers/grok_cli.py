"""Grok Build CLI provider (xAI).

Drives the real ``grok`` binary inside a tmux pane so the user's xAI /
SuperGrok subscription and native Grok Build features (Plan Mode via Shift+Tab,
worktrees, subagents) are preserved. We never pass an API key; auth is inherited
from the user's existing ``~/.grok`` login.

Markers below were captured from a LIVE ``grok`` session (grok 0.2.14-mac):
  * ready/idle footer  : "Shift+Tab:mode    Ctrl+.:shortcuts"
  * mode bar (always)  : "Grok Build  always-approve"   (default = auto-approve)
  * turn complete      : "Turn completed in <n>s."
  * thinking line      : "Thought for <n>s"
  * user turn lines are right-stamped with a clock time (e.g. "6:18 PM")

Grok renders on the ALTERNATE screen (alt=1), so we read the visible viewport
via capture-pane, not scrollback. ``always-approve`` is the default mode, so
tool use does not block on permission prompts — good for orchestration.

Template: providers/codex.py. Status ordering mirrors Codex's hard-won rule —
recognize work-in-flight BEFORE concluding COMPLETED, or a mid-turn frame is
misread as done.
"""

import re
from typing import List, Optional

from cli_agent_orchestrator.clients.tmux import tmux_client
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.base import BaseProvider
from cli_agent_orchestrator.utils.terminal import wait_for_shell, wait_until_status

# --- Verified Grok TUI markers (grok 0.2.14-mac) ---------------------------
# Idle footer present whenever Grok is ready for input.
IDLE_FOOTER_PATTERN = re.compile(r"Shift\+Tab:\s*mode|Ctrl\+\.\s*:\s*shortcuts")
# Per-turn completion line.
TURN_COMPLETE_PATTERN = re.compile(r"Turn completed in\s+[\d.]+s")
# Work-in-flight hints (spinner is brief; these catch active turns/plans).
PROCESSING_PATTERN = re.compile(
    r"Esc to interrupt|Thinking|Working|Generating|Running|esc to interrupt",
    re.IGNORECASE,
)
# Plan Mode / subagent activity — also PROCESSING (never COMPLETED mid-plan).
# TODO(grok): confirm exact Plan Mode banner string when exercised live.
PLAN_OR_SUBAGENT_PATTERN = re.compile(
    r"Plan mode|Planning|subagent|sub-agent|worktree task|delegating",
    re.IGNORECASE,
)
# Approval prompt. Default mode is "always-approve" so this is rare, but a
# non-default mode can still prompt. Patterns require explicit yes/no
# AFFORDANCES rather than the bare word "approve" — the permanent mode bar
# literally reads "Grok Build  always-approve", and a naive \bApprove\b would
# match the "approve" inside "always-approve" and falsely report WAITING on
# every idle/completed frame. TODO(grok): pin exact approval strings live.
WAITING_PROMPT_PATTERN = re.compile(
    r"\(y/N\)|\(Y/n\)|\by/n\b|\byes/no\b|"
    r"(?:Allow|Approve|Proceed|Apply this change)\b[^\n]*\?",
    re.IGNORECASE,
)
# Lines that are permanent TUI chrome (mode bar / footer) — excluded from the
# WAITING check so their static text can never trigger a false prompt match.
CHROME_LINE_PATTERN = re.compile(
    r"always-approve|Grok Build|Shift\+Tab:|Ctrl\+\."
)
ERROR_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Error:|ERROR:|panic:|rate limit|quota exceeded|"
    r"authentication failed|failed to)",
    re.IGNORECASE,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[\[\]()=><][0-9;?]*\x07?")

# Bottom-N visible lines holding the footer/mode bar on the alt-screen.
FOOTER_TAIL_LINES = 6
# Enough of the alt-screen viewport for status + extraction.
STATUS_TAIL_LINES = 160


class GrokCliProvider(BaseProvider):
    """Provider for the Grok Build CLI (interactive TUI, primary path)."""

    def __init__(
        self,
        terminal_id: str,
        session_name: str,
        window_name: str,
        agent_profile: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        skill_prompt: Optional[str] = None,
    ):
        # Arg ORDER mirrors CodexProvider exactly — manager.create_provider calls
        # providers positionally: (tid, session, window, agent_profile, allowed_tools, ...).
        super().__init__(
            terminal_id, session_name, window_name, allowed_tools, skill_prompt
        )
        # Agent profile NAME (str), loaded lazily in _build_grok_command().
        self._agent_profile = agent_profile
        self._initialized = False

    @property
    def paste_enter_count(self) -> int:
        # All providers except Claude Code submit on a single Enter after
        # bracketed paste. Verified: grok submits on one Enter.
        return 1

    def _build_grok_command(self) -> str:
        """Build the interactive launch command.

        Auth is inherited from ~/.grok — we deliberately do NOT pass a key, so
        the user's xAI / SuperGrok subscription is preserved. ``--always-approve``
        auto-approves tool use so orchestration is not blocked by permission
        prompts. We keep the alternate screen (Grok's default) because its
        status markers are cleaner there than inline. The model is honored when
        the agent profile declares one.

        Verified flags (grok 0.2.14 --help): --always-approve, -m/--model,
        --agent <file>, --allow/--deny, --no-plan, --no-subagents, --cwd.
        """
        import shlex

        parts = ["grok", "--always-approve"]

        if self._agent_profile is not None:
            try:
                from cli_agent_orchestrator.utils.agent_profiles import (
                    load_agent_profile,
                )

                profile = load_agent_profile(self._agent_profile)
                if getattr(profile, "model", None):
                    parts.extend(["--model", str(profile.model)])
            except Exception:
                # Profile is optional; a missing/invalid one must not block launch.
                pass

        # NOTE: Grok exposes --agent <file> for agent definitions, but no simple
        # inline system-prompt flag; profile instructions are not injected in v1.
        # Model + subscription behavior ARE preserved.
        # TODO(grok): write the profile system_prompt to a temp --agent file.
        return shlex.join(parts)

    def initialize(self) -> bool:
        """Launch grok in the pane and wait until it is ready for input."""
        import time

        if not wait_for_shell(
            tmux_client, self.session_name, self.window_name, timeout=10.0
        ):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        # Warm-up like codex: TUIs can exit immediately in a freshly-created
        # tmux pane before the shell finishes its first interactive cycle.
        tmux_client.send_keys(self.session_name, self.window_name, "echo ready")
        time.sleep(2.0)

        command = self._build_grok_command()
        tmux_client.send_keys(self.session_name, self.window_name, command)

        # Grok boots into the alt-screen; the idle footer appears when ready.
        ready = wait_until_status(
            self,
            {TerminalStatus.IDLE, TerminalStatus.COMPLETED},
            timeout=60.0,
            polling_interval=1.0,
        )
        self._initialized = ready
        return ready

    def _capture(self, tail_lines: Optional[int]) -> str:
        raw = tmux_client.get_history(
            self.session_name,
            self.window_name,
            tail_lines=tail_lines or STATUS_TAIL_LINES,
            strip_escapes=True,
        )
        return ANSI_RE.sub("", raw)

    def get_status(self, tail_lines: Optional[int] = None) -> TerminalStatus:
        """Classify the visible Grok viewport into a TerminalStatus.

        Order matters: work-in-flight (spinner / Plan Mode / subagent) must be
        recognized BEFORE concluding COMPLETED, or a mid-turn frame is misread.
        """
        screen = self._capture(tail_lines)
        if not screen.strip():
            return TerminalStatus.PROCESSING

        nonblank = [ln for ln in screen.splitlines() if ln.strip()]
        tail = "\n".join(nonblank[-FOOTER_TAIL_LINES:])
        has_footer = bool(IDLE_FOOTER_PATTERN.search(tail))

        # 1) Approval/permission dialog needs the user. Check ONLY non-chrome
        #    lines so the permanent "always-approve" mode bar can't false-match.
        prompt_area = "\n".join(
            ln for ln in nonblank[-FOOTER_TAIL_LINES:]
            if not CHROME_LINE_PATTERN.search(ln)
        )
        if WAITING_PROMPT_PATTERN.search(prompt_area):
            return TerminalStatus.WAITING_USER_ANSWER

        # 2) Errors.
        if ERROR_PATTERN.search(screen):
            return TerminalStatus.ERROR

        # 3) Work in flight (spinner / Plan Mode / subagent) — BEFORE completion.
        if PROCESSING_PATTERN.search(screen) or PLAN_OR_SUBAGENT_PATTERN.search(
            screen
        ):
            return TerminalStatus.PROCESSING

        # 4) Turn-complete marker present → COMPLETED. The "Turn completed in Ns."
        #    line only appears after a real turn finishes, so the marker alone is
        #    sufficient — do NOT gate on self._initialized: the status poller
        #    creates providers on-demand (init flag False) and would otherwise
        #    never report COMPLETED.
        if TURN_COMPLETE_PATTERN.search(screen):
            return TerminalStatus.COMPLETED

        # 5) Idle footer visible, nothing in flight → ready for input.
        if has_footer:
            return TerminalStatus.IDLE

        # 6) Nothing conclusive — assume work in progress.
        return TerminalStatus.PROCESSING

    def get_idle_pattern_for_log(self) -> str:
        # Present in the raw pane stream when idle.
        return r"Shift\+Tab:\s*mode"

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Best-effort extraction of Grok's last reply.

        Returns text after the last user turn, stripped of TUI chrome and the
        completion/footer lines. Good enough for review display.
        TODO(grok): tighten with a confirmed assistant prefix.
        """
        text = ANSI_RE.sub("", script_output)
        lines = [ln.rstrip() for ln in text.splitlines()]

        chrome = re.compile(
            r"Shift\+Tab:\s*mode|Ctrl\+\.\s*:\s*shortcuts|Grok Build|"
            r"always-approve|Turn completed in|Thought for|New worktree|"
            r"Resume session|^\s*Quit\s|Tip:|Beta\s*$"
        )
        body = [ln for ln in lines if ln.strip() and not chrome.search(ln)]
        return "\n".join(body[-200:]).strip()

    def exit_cli(self) -> str:
        # Verified footer: "Quit  ctrl-q". Ctrl-q cleanly exits the TUI.
        return "C-q"

    def cleanup(self) -> None:
        self._initialized = False
