"""
DropZone widget providing a visually distinct Drag & Drop area for PDF files.
"""

import customtkinter as ctk
from pathlib import Path
from typing import List, Callable, Optional
from ..theme import Theme

# Check if tkinterdnd2 is available
try:
    from tkinterdnd2 import DND_FILES
    HAS_TKDND = True
except ImportError:
    HAS_TKDND = False


class DropZone(ctk.CTkFrame):
    """Custom Drag & Drop target container with hover feedback."""

    def __init__(
        self,
        master: any,
        on_files_dropped: Callable[[List[Path]], None],
        on_click_browse: Callable[[], None],
        **kwargs
    ):
        super().__init__(
            master,
            fg_color=Theme.SURFACE_VARIANT,
            border_color=Theme.BORDER_LIGHT,
            border_width=2,
            corner_radius=Theme.RADIUS_MD,
            **kwargs
        )
        self.on_files_dropped = on_files_dropped
        self.on_click_browse = on_click_browse

        self._build_ui()
        self._setup_events()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        self.icon_label = ctk.CTkLabel(
            self,
            text="📄",
            font=("Segoe UI", 36),
            text_color=Theme.PRIMARY
        )
        self.icon_label.pack(pady=(20, 5))

        self.title_label = ctk.CTkLabel(
            self,
            text="Drop PDFs Here",
            font=Theme.FONT_SUBTITLE,
            text_color=Theme.TEXT_PRIMARY
        )
        self.title_label.pack(pady=2)

        self.sub_label = ctk.CTkLabel(
            self,
            text="or click to browse local files",
            font=Theme.FONT_SMALL,
            text_color=Theme.TEXT_SECONDARY
        )
        self.sub_label.pack(pady=(0, 20))

    def _setup_events(self) -> None:
        # Hover animations
        self.bind("<Enter>", self._on_hover_enter)
        self.bind("<Leave>", self._on_hover_leave)
        self.bind("<Button-1>", lambda e: self.on_click_browse())

        for widget in (self.icon_label, self.title_label, self.sub_label):
            widget.bind("<Enter>", self._on_hover_enter)
            widget.bind("<Leave>", self._on_hover_leave)
            widget.bind("<Button-1>", lambda e: self.on_click_browse())

        # Register TkDND drag and drop if available
        if HAS_TKDND:
            try:
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self._on_dnd_drop)
            except Exception:
                pass

    def _on_hover_enter(self, event=None) -> None:
        self.configure(
            fg_color=Theme.SURFACE_HOVER,
            border_color=Theme.PRIMARY
        )

    def _on_hover_leave(self, event=None) -> None:
        self.configure(
            fg_color=Theme.SURFACE_VARIANT,
            border_color=Theme.BORDER_LIGHT
        )

    def _on_dnd_drop(self, event) -> None:
        self._on_hover_leave()
        if not event.data:
            return

        # Parse drop paths (TkDND format handles space-enclosed curly braces)
        raw_data = event.data
        file_paths = []

        if raw_data.startswith('{'):
            # Split paths enclosed in braces
            import re
            file_paths = [re.sub(r'[\{\}]', '', item) for item in re.findall(r'\{[^\}]+\}|[^\s]+', raw_data)]
        else:
            file_paths = raw_data.split()

        valid_paths = [Path(p) for p in file_paths if p.lower().endswith('.pdf')]
        if valid_paths and self.on_files_dropped:
            self.on_files_dropped(valid_paths)
