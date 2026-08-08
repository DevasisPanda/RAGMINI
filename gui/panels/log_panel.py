"""
LogPanel providing live, color-coded, auto-scrolling terminal log output inside the GUI.
"""

import logging
import queue
import customtkinter as ctk
from ..theme import Theme


class QueueLogHandler(logging.Handler):
    """Custom logging handler routing Python log records to a thread-safe Queue."""

    def __init__(self, message_queue: queue.Queue):
        super().__init__()
        self.message_queue = message_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.message_queue.put(("log", (record.levelno, msg)))
        except Exception:
            self.handleError(record)


class LogPanel(ctk.CTkFrame):
    """Panel containing a scrollable read-only log window."""

    def __init__(self, master: any, max_lines: int = 500, **kwargs):
        super().__init__(
            master,
            fg_color=Theme.SURFACE,
            border_color=Theme.BORDER,
            border_width=1,
            corner_radius=Theme.RADIUS_MD,
            **kwargs
        )
        self.max_lines = max_lines
        self.line_count = 0

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header Row
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=Theme.PAD_MD, pady=(Theme.PAD_MD, Theme.PAD_XS))
        header_frame.grid_columnconfigure(0, weight=1)

        lbl_title = ctk.CTkLabel(
            header_frame,
            text="Pipeline Diagnostics Log",
            font=Theme.FONT_TITLE,
            text_color=Theme.TEXT_PRIMARY
        )
        lbl_title.grid(row=0, column=0, sticky="w")

        btn_clear_log = ctk.CTkButton(
            header_frame,
            text="Clear Log",
            font=Theme.FONT_SMALL,
            fg_color="transparent",
            hover_color=Theme.SURFACE_HOVER,
            text_color=Theme.TEXT_SECONDARY,
            width=70,
            height=24,
            command=self.clear_log
        )
        btn_clear_log.grid(row=0, column=1, sticky="e")

        # Textbox log viewer
        self.textbox = ctk.CTkTextbox(
            self,
            font=Theme.FONT_CODE,
            fg_color=Theme.BG_DARK,
            text_color=Theme.TEXT_PRIMARY,
            border_color=Theme.BORDER_LIGHT,
            border_width=1,
            corner_radius=Theme.RADIUS_SM,
            wrap="word",
            activate_scrollbars=True
        )
        self.textbox.grid(row=1, column=0, sticky="nsew", padx=Theme.PAD_MD, pady=(0, Theme.PAD_MD))

        # Initial greeting line
        self.append_log(logging.INFO, "[INFO] TinyRAG Desktop Logging system initialized.")

    def append_log(self, levelno: int, text: str) -> None:
        """Append a log line to the viewer with color tagging and auto-scroll."""
        self.textbox.configure(state="normal")

        # Format line with prefix symbol
        prefix = ""
        if levelno >= logging.ERROR:
            prefix = "[ERROR] "
        elif levelno >= logging.WARNING:
            prefix = "[WARN] "
        elif "[OK]" in text:
            prefix = ""
        elif levelno >= logging.INFO:
            prefix = ""

        formatted_line = f"{text}\n" if text.endswith("\n") is False else text
        self.textbox.insert("end", formatted_line)

        self.line_count += 1
        if self.line_count > self.max_lines:
            # Delete oldest line to keep buffer size bounded
            self.textbox.delete("1.0", "2.0")
            self.line_count -= 1

        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def clear_log(self) -> None:
        """Clear all log entries."""
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.configure(state="disabled")
        self.line_count = 0
