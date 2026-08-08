"""
UploadPanel for managing PDF selection, Google Drive downloads, and indexing controls.
"""

import customtkinter as ctk
from pathlib import Path
from typing import List, Callable, Optional

from ..theme import Theme
from ..widgets.drop_zone import DropZone
from ..widgets.animated_button import AnimatedButton


class UploadPanel(ctk.CTkFrame):
    """Panel containing file drag & drop, drive link download, file list, and indexing controls."""

    def __init__(
        self,
        master: any,
        on_browse_clicked: Callable[[], None],
        on_clear_clicked: Callable[[], None],
        on_start_indexing: Callable[[], None],
        on_reindex_clicked: Callable[[], None],
        on_drive_download: Callable[[str], None],
        on_files_dropped: Callable[[List[Path]], None],
        on_file_removed: Callable[[Path], None],
        **kwargs
    ):
        super().__init__(
            master,
            fg_color=Theme.SURFACE,
            border_color=Theme.BORDER,
            border_width=1,
            corner_radius=Theme.RADIUS_MD,
            **kwargs
        )
        self.on_browse_clicked = on_browse_clicked
        self.on_clear_clicked = on_clear_clicked
        self.on_start_indexing = on_start_indexing
        self.on_reindex_clicked = on_reindex_clicked
        self.on_drive_download = on_drive_download
        self.on_files_dropped = on_files_dropped
        self.on_file_removed = on_file_removed

        self.selected_files: List[Path] = []

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        # Header Title
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.pack(fill="x", padx=Theme.PAD_MD, pady=(Theme.PAD_MD, Theme.PAD_SM))

        lbl_title = ctk.CTkLabel(
            title_frame,
            text="Document Ingestion",
            font=Theme.FONT_TITLE,
            text_color=Theme.TEXT_PRIMARY
        )
        lbl_title.pack(side="left")

        # 1. Drag & Drop Zone
        self.drop_zone = DropZone(
            self,
            on_files_dropped=self.on_files_dropped,
            on_click_browse=self.on_browse_clicked
        )
        self.drop_zone.pack(fill="x", padx=Theme.PAD_MD, pady=Theme.PAD_XS)

        # 2. Google Drive Link Bar
        drive_frame = ctk.CTkFrame(self, fg_color="transparent")
        drive_frame.pack(fill="x", padx=Theme.PAD_MD, pady=Theme.PAD_SM)
        drive_frame.grid_columnconfigure(0, weight=1)

        self.drive_entry = ctk.CTkEntry(
            drive_frame,
            placeholder_text="Paste Google Drive sharing link...",
            font=Theme.FONT_BODY,
            height=34,
            border_color=Theme.BORDER_LIGHT,
            fg_color=Theme.SURFACE_VARIANT
        )
        self.drive_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.drive_entry.bind("<Return>", lambda e: self._handle_drive_download())

        self.btn_drive = AnimatedButton(
            drive_frame,
            text="Fetch Link",
            command=self._handle_drive_download,
            variant="secondary",
            icon="☁️",
            width=100,
            height=34
        )
        self.btn_drive.grid(row=0, column=1, sticky="e")

        # 3. Selected Files Scrollable Frame
        self.file_list_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=Theme.SURFACE_VARIANT,
            height=85,
            corner_radius=Theme.RADIUS_SM
        )
        self.file_list_frame.pack(fill="x", padx=Theme.PAD_MD, pady=Theme.PAD_XS)
        self.file_list_frame.grid_columnconfigure(0, weight=1)

        # 4. Progress Bar
        self.progress_bar = ctk.CTkProgressBar(
            self,
            progress_color=Theme.PRIMARY,
            fg_color=Theme.SURFACE_VARIANT,
            height=8
        )
        self.progress_bar.pack(fill="x", padx=Theme.PAD_MD, pady=(Theme.PAD_SM, 0))
        self.progress_bar.set(0.0)

        # 5. Buttons Row
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=Theme.PAD_MD, pady=Theme.PAD_MD)
        btn_row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.btn_browse = AnimatedButton(
            btn_row, text="Browse", command=self.on_browse_clicked, variant="secondary", icon="📁"
        )
        self.btn_browse.grid(row=0, column=0, padx=2, sticky="ew")

        self.btn_clear = AnimatedButton(
            btn_row, text="Clear", command=self.on_clear_clicked, variant="ghost", icon="🗑️"
        )
        self.btn_clear.grid(row=0, column=1, padx=2, sticky="ew")

        self.btn_index = AnimatedButton(
            btn_row, text="Start Indexing", command=self.on_start_indexing, variant="primary", icon="▶"
        )
        self.btn_index.grid(row=0, column=2, padx=2, sticky="ew")

        self.btn_reindex = AnimatedButton(
            btn_row, text="Re-Index", command=self.on_reindex_clicked, variant="secondary", icon="🔄"
        )
        self.btn_reindex.grid(row=0, column=3, padx=2, sticky="ew")

    def _handle_drive_download(self) -> None:
        url = self.drive_entry.get().strip()
        if url and self.on_drive_download:
            self.on_drive_download(url)
            self.drive_entry.delete(0, "end")

    def update_file_list(self, files: List[Path]) -> None:
        """Refresh the selected files list widget."""
        self.selected_files = list(files)

        # Clear existing list items
        for child in self.file_list_frame.winfo_children():
            child.destroy()

        if not files:
            lbl_empty = ctk.CTkLabel(
                self.file_list_frame,
                text="No PDFs added yet",
                font=Theme.FONT_SMALL,
                text_color=Theme.TEXT_SECONDARY
            )
            lbl_empty.pack(pady=20)
            return

        for idx, fpath in enumerate(files):
            item_frame = ctk.CTkFrame(self.file_list_frame, fg_color="transparent")
            item_frame.pack(fill="x", pady=2)
            item_frame.grid_columnconfigure(1, weight=1)

            icn = ctk.CTkLabel(item_frame, text="📄", font=Theme.FONT_SMALL)
            icn.grid(row=0, column=0, padx=(4, 6))

            name_lbl = ctk.CTkLabel(
                item_frame,
                text=fpath.name,
                font=Theme.FONT_SMALL,
                text_color=Theme.TEXT_PRIMARY,
                anchor="w"
            )
            name_lbl.grid(row=0, column=1, sticky="w")

            btn_del = ctk.CTkButton(
                item_frame,
                text="✕",
                width=24,
                height=20,
                fg_color="transparent",
                hover_color=Theme.ERROR,
                text_color=Theme.TEXT_SECONDARY,
                command=lambda p=fpath: self.on_file_removed(p)
            )
            btn_del.grid(row=0, column=2, padx=4)

    def set_progress(self, value: float) -> None:
        """Set progress bar value (0.0 to 1.0)."""
        self.progress_bar.set(max(0.0, min(1.0, value)))

    def set_indexing_state(self, is_indexing: bool) -> None:
        """Toggle controls disabled state during indexing."""
        if is_indexing:
            self.btn_index.set_loading(True, "Indexing...")
            self.btn_reindex.configure(state="disabled")
            self.btn_browse.configure(state="disabled")
            self.btn_clear.configure(state="disabled")
            self.btn_drive.configure(state="disabled")
        else:
            self.btn_index.set_loading(False)
            self.btn_reindex.configure(state="normal")
            self.btn_browse.configure(state="normal")
            self.btn_clear.configure(state="normal")
            self.btn_drive.configure(state="normal")
