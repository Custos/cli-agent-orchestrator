"""Grok one-shot (headless) fast path.

A convenience, NOT the default. The interactive TUI provider
(``providers/grok_cli.py``) is the primary path; this runs ``grok -p`` for a
quick single-prompt answer without a long-lived tmux pane. It does not flow
through the provider/get_status machinery.

Auth is inherited from the user's ``~/.grok`` login — we never pass an API key,
so the xAI / SuperGrok subscription is preserved.
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GrokOneshotResult:
    ok: bool
    output: str
    error: Optional[str] = None


def run_grok_oneshot(
    prompt: str,
    working_directory: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 120.0,
) -> GrokOneshotResult:
    """Run ``grok -p <prompt>`` and return stdout.

    Args:
        prompt: The single prompt to send.
        working_directory: Optional cwd (also passed as grok's --cwd).
        model: Optional model override (grok's -m/--model flag).
        timeout: Hard wall-clock cap in seconds.
    """
    if not shutil.which("grok"):
        return GrokOneshotResult(False, "", "grok binary not found on PATH")

    # -p/--prompt runs headless; --always-approve avoids interactive gating.
    cmd = ["grok", "--always-approve", "-p", prompt]
    if model:
        cmd.extend(["--model", model])
    if working_directory:
        cmd.extend(["--cwd", working_directory])

    try:
        proc = subprocess.run(
            cmd,
            cwd=working_directory or None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return GrokOneshotResult(False, "", f"grok timed out after {timeout}s")
    except Exception as e:  # pragma: no cover - defensive
        return GrokOneshotResult(False, "", str(e))

    if proc.returncode != 0:
        return GrokOneshotResult(
            False,
            proc.stdout or "",
            (proc.stderr or "").strip() or f"grok exited {proc.returncode}",
        )
    return GrokOneshotResult(True, proc.stdout, None)
