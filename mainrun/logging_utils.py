import json
import time
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm


class DualLogger:
    """Write structured JSON events to a file and readable lines to stdout."""

    def __init__(self, file_handler: TextIO) -> None:
        self.file_handler = file_handler

    def log(self, event: str, **kwargs: Any) -> None:
        """Append one JSON event line to the log file and optionally print a summary."""
        log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
        self.file_handler.write(log_entry + "\n")
        self.file_handler.flush()

        if not kwargs.get("prnt", True):
            return
        if "step" in kwargs and "max_steps" in kwargs:
            tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
        else:
            parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
            tqdm.write(f"{event}: {', '.join(parts)}" if parts else event)


def configure_logging(log_file: str) -> DualLogger:
    """Create the log file's parent directory and return a DualLogger writing to it."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = open(log_file, "w")
    return DualLogger(file_handler)
