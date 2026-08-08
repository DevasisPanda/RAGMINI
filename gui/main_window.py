"""
MainWindow providing grid layout management, event routing, and queue polling loop.
"""

import queue
import logging
from pathlib import Path
from typing import List
import customtkinter as ctk

# Import TkDND if installed
try:
    from tkinterdnd2 import TkinterDnD
    class CTkWithDnD(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
    HAS_TKDND = True
except Exception:
    CTkWithDnD = ctk.CTk
    HAS_TKDND = False

from .theme import Theme
from .panels.status_bar import StatusBar
from .panels.upload_panel import UploadPanel
from .panels.log_panel import LogPanel, QueueLogHandler
from .panels.chat_panel import ChatPanel
from controllers.backend_controller import BackendController

logger = logging.getLogger(__name__)


class MainWindow(CTkWithDnD):
    """Main Application Window."""

    def __init__(self, controller: BackendController):
        super().__init__()
        self.controller = controller

        # Set window properties
        self.title("AssignRAG Desktop v2.0 — Local PDF Question Answering System")
        self.geometry("1100x760")
        self.minsize(960, 650)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Configure root grid weights
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=0)  # Status Bar
        self.grid_rowconfigure(1, weight=4)  # Middle Section (Upload + Log)
        self.grid_rowconfigure(2, weight=5)  # Bottom Section (Chat)

        self._build_panels()
        self._setup_log_redirection()
        self._start_queue_polling()

        # Auto-load existing PDFs in data/test_pdfs or data/pdfs if available
        self._load_initial_pdfs()

    def _build_panels(self) -> None:
        # Top: Status Bar
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

        # Update initial provider info in status bar
        p_name, m_name = self.controller.get_provider_display_info()
        self.status_bar.set_provider_info(p_name, m_name)

        # Middle Left: Upload Panel
        self.upload_panel = UploadPanel(
            self,
            on_browse_clicked=self._on_browse_pdfs,
            on_clear_clicked=self._on_clear_pdfs,
            on_start_indexing=self._on_start_indexing,
            on_reindex_clicked=self._on_reindex_pdfs,
            on_drive_download=self._on_drive_download,
            on_files_dropped=self._on_files_dropped,
            on_file_removed=self._on_file_removed
        )
        self.upload_panel.grid(row=1, column=0, sticky="nsew", padx=(Theme.PAD_MD, Theme.PAD_SM), pady=Theme.PAD_SM)

        # Middle Right: Log Panel
        self.log_panel = LogPanel(self)
        self.log_panel.grid(row=1, column=1, sticky="nsew", padx=(Theme.PAD_SM, Theme.PAD_MD), pady=Theme.PAD_SM)

        # Bottom: Chat Panel
        self.chat_panel = ChatPanel(
            self,
            on_ask_submitted=self._on_ask_submitted,
            on_clear_chat=self._on_clear_chat
        )
        self.chat_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=Theme.PAD_MD, pady=(Theme.PAD_SM, Theme.PAD_MD))

    def _setup_log_redirection(self) -> None:
        """Attach custom logging handler to route backend log messages into GUI log panel."""
        gui_handler = QueueLogHandler(self.controller.message_queue)
        gui_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(gui_handler)

    def _start_queue_polling(self) -> None:
        """Poll the thread-safe message queue every 100ms for UI events."""
        try:
            while True:
                event_type, data = self.controller.message_queue.get_nowait()
                self._handle_queue_event(event_type, data)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._start_queue_polling)

    def _handle_queue_event(self, event_type: str, data: any) -> None:
        """Dispatch UI events received from background threads."""
        if event_type == "status":
            state, msg = data
            self.status_bar.set_status(state, msg)
            if state == "indexing":
                self.upload_panel.set_indexing_state(True)
            elif state == "thinking":
                self.chat_panel.set_thinking_state(True)
            elif state == "ready":
                self.upload_panel.set_indexing_state(False)
                self.chat_panel.set_thinking_state(False)
            elif state == "error":
                self.upload_panel.set_indexing_state(False)
                self.chat_panel.set_thinking_state(False)

        elif event_type == "progress":
            self.upload_panel.set_progress(data)

        elif event_type == "log":
            levelno, msg = data
            self.log_panel.append_log(levelno, msg)

        elif event_type == "index_complete":
            self.upload_panel.set_indexing_state(False)
            self.upload_panel.set_progress(1.0)

        elif event_type == "answer":
            self.chat_panel.set_thinking_state(False)
            self.chat_panel.display_query_result(data)

        elif event_type == "warning":
            self.log_panel.append_log(logging.WARNING, f"[WARN] {data}")

        elif event_type == "error":
            self.log_panel.append_log(logging.ERROR, f"[ERROR] {data}")

    def _load_initial_pdfs(self) -> None:
        """Locate default test PDFs in workspace if present."""
        test_dir = Path("data/test_pdfs")
        if not test_dir.exists():
            test_dir = Path("data/pdfs")

        if test_dir.exists():
            pdf_files = list(test_dir.glob("*.pdf"))
            if pdf_files:
                self.controller.file_manager.add_files(pdf_files)
                self.upload_panel.update_file_list(self.controller.file_manager.get_selected_files())
                self.status_bar.set_doc_count(len(self.controller.file_manager.get_selected_files()))

    # Event Callbacks
    def _on_browse_pdfs(self) -> None:
        file_paths = ctk.filedialog.askopenfilenames(
            title="Select PDF Documents",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_paths:
            paths = [Path(p) for p in file_paths]
            added, warnings = self.controller.file_manager.add_files(paths)
            self.upload_panel.update_file_list(self.controller.file_manager.get_selected_files())
            self.status_bar.set_doc_count(len(self.controller.file_manager.get_selected_files()))
            for warn in warnings:
                self.log_panel.append_log(logging.WARNING, f"[WARN] {warn}")

    def _on_files_dropped(self, dropped_paths: List[Path]) -> None:
        added, warnings = self.controller.file_manager.add_files(dropped_paths)
        self.upload_panel.update_file_list(self.controller.file_manager.get_selected_files())
        self.status_bar.set_doc_count(len(self.controller.file_manager.get_selected_files()))
        for warn in warnings:
            self.log_panel.append_log(logging.WARNING, f"[WARN] {warn}")

    def _on_file_removed(self, path: Path) -> None:
        self.controller.file_manager.remove_file(path)
        self.upload_panel.update_file_list(self.controller.file_manager.get_selected_files())
        self.status_bar.set_doc_count(len(self.controller.file_manager.get_selected_files()))

    def _on_clear_pdfs(self) -> None:
        self.controller.file_manager.clear_files()
        self.upload_panel.update_file_list([])
        self.upload_panel.set_progress(0.0)
        self.status_bar.set_doc_count(0)
        self.log_panel.append_log(logging.INFO, "[INFO] Cleared PDF selection.")

    def _on_drive_download(self, drive_url: str) -> None:
        self.log_panel.append_log(logging.INFO, f"[INFO] Attempting Google Drive download from link...")
        path, msg = self.controller.file_manager.download_google_drive_pdf(drive_url)
        if path:
            self.controller.file_manager.add_files([path])
            self.upload_panel.update_file_list(self.controller.file_manager.get_selected_files())
            self.status_bar.set_doc_count(len(self.controller.file_manager.get_selected_files()))
            self.log_panel.append_log(logging.INFO, f"[OK] {msg}")
        else:
            self.log_panel.append_log(logging.ERROR, f"[ERROR] {msg}")

    def _on_start_indexing(self) -> None:
        selected = self.controller.file_manager.get_selected_files()
        if not selected:
            self.log_panel.append_log(logging.WARNING, "[WARN] No PDF files selected for indexing.")
            return

        self.log_panel.append_log(logging.INFO, f"[INFO] Starting indexing for {len(selected)} PDF document(s)...")
        self.controller.index_pdfs_async(selected, recreate_collection=True)

    def _on_reindex_pdfs(self) -> None:
        selected = self.controller.file_manager.get_selected_files()
        if not selected:
            self.log_panel.append_log(logging.WARNING, "[WARN] No PDF files selected to re-index.")
            return

        self.log_panel.append_log(logging.INFO, f"[INFO] Re-indexing collection with {len(selected)} PDF document(s)...")
        self.controller.index_pdfs_async(selected, recreate_collection=True)

    def _on_ask_submitted(self, question: str) -> None:
        self.log_panel.append_log(logging.INFO, f"[INFO] User Query: '{question}'")
        self.controller.ask_question_async(question)

    def _on_clear_chat(self) -> None:
        self.log_panel.append_log(logging.INFO, "[INFO] Cleared chat history.")
