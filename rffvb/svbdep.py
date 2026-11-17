"""Read-only use of georgepaul171/2027-research (svb package): run_cavi."""
import subprocess
import sys

from . import config as C

if C.SVB_PATH not in sys.path:
    sys.path.insert(0, C.SVB_PATH)
from svb.core import run_cavi            # noqa: E402


def svb_commit():
    try:
        return subprocess.run(["git", "-C", C.SVB_PATH, "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return "unknown"
