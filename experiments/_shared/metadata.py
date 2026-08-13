"""Environment and provenance helpers shared by experiment scripts."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Optional


def command_output(command: List[str], timeout: int = 30) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return result.stderr.strip() or result.stdout.strip() or None
    except Exception:
        return None


def package_version(package: str) -> Optional[str]:
    code = (
        "import importlib.metadata as m\n"
        f"pkg={package!r}\n"
        "try:\n"
        "    print(m.version(pkg))\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__ + ': ' + str(exc))\n"
    )
    return command_output([sys.executable, "-c", code])


def resolved_revision(repo_id: str, repo_type: str, revision: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
        return getattr(info, "sha", None)
    except Exception:
        return revision


def created_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
