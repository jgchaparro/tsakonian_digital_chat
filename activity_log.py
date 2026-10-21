"""Activity log: what the chat did, on the console and in logs/activity.log."""

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE = Path(__file__).with_name("logs") / "activity.log"
log = logging.getLogger("tsakonian_chat")
# Without setup_logging (tests, imports) records are dropped, not printed to stderr.
log.addHandler(logging.NullHandler())


def setup_logging():
    """Attach the handlers once; Streamlit reruns the script on every interaction."""
    if any(isinstance(handler, RotatingFileHandler) for handler in log.handlers):
        return
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    # Greek and Tsakonian text must not break a console with a legacy code page.
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    LOG_FILE.parent.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    for handler in (file_handler, logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(handler)
    log.propagate = False


def clip(text, limit=500):
    """One quoted line of a message for the log, cut when very long."""
    shown = json.dumps(text, ensure_ascii=False)
    return shown if len(shown) <= limit else f"{shown[:limit]}… ({len(text)} chars)"
