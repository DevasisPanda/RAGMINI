"""
CitationCard widget for displaying structured sources with expandable snippets.
"""

import customtkinter as ctk
from typing import List
from ..theme import Theme
from Rag.core.models import Citation


class CitationCard(ctk.CTkFrame):
    """Card widget representing supporting document citations."""

    def __init__(
        self,
        master: any,
        document_name: str,
        pages: List[int],
        citations: List[Citation],
        **kwargs
    ):
        super().__init__(
            master,
            fg_color=Theme.SURFACE_VARIANT,
            border_color=Theme.BORDER,
            border_width=1,
            corner_radius=Theme.RADIUS_SM,
            **kwargs
        )
        self.document_name = document_name
        self.pages = sorted(list(set(pages)))
        self.citations = citations
        self._is_expanded = False

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        # Header Row
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=6)
        self.header_frame.grid_columnconfigure(1, weight=1)

        doc_icon = ctk.CTkLabel(
            self.header_frame,
            text="📄",
            font=("Segoe UI", 14)
        )
        doc_icon.grid(row=0, column=0, padx=(0, 6), sticky="w")

        title_label = ctk.CTkLabel(
            self.header_frame,
            text=self.document_name,
            font=Theme.FONT_SUBTITLE,
            text_color=Theme.TEXT_PRIMARY,
            anchor="w"
        )
        title_label.grid(row=0, column=1, sticky="w")

        pages_str = "Pages: " + ", ".join(str(p) for p in self.pages)
        pages_badge = ctk.CTkLabel(
            self.header_frame,
            text=pages_str,
            font=Theme.FONT_SMALL,
            text_color=Theme.PRIMARY,
            fg_color=Theme.SURFACE_HOVER,
            corner_radius=4,
            padx=8,
            pady=2
        )
        pages_badge.grid(row=0, column=2, padx=6, sticky="e")

        self.expand_btn = ctk.CTkButton(
            self.header_frame,
            text="▼ Show Text",
            font=Theme.FONT_SMALL,
            fg_color="transparent",
            text_color=Theme.TEXT_SECONDARY,
            hover_color=Theme.SURFACE_HOVER,
            width=70,
            height=24,
            command=self._toggle_expand
        )
        self.expand_btn.grid(row=0, column=3, sticky="e")

        # Collapsible Details Frame
        self.details_frame = ctk.CTkFrame(self, fg_color="transparent")

        for idx, citation in enumerate(self.citations, 1):
            snippet = citation.retrieved_text[:280].replace("\n", " ").strip()
            score_str = f"Score: {citation.similarity_score:.3f}"

            item_lbl = ctk.CTkLabel(
                self.details_frame,
                text=f"• Page {citation.page_number} ({score_str}):\n  \"{snippet}...\"",
                font=Theme.FONT_CODE,
                text_color=Theme.TEXT_SECONDARY,
                anchor="w",
                justify="left",
                wraplength=600
            )
            item_lbl.pack(fill="x", padx=12, pady=4)

    def _toggle_expand(self) -> None:
        self._is_expanded = not self._is_expanded
        if self._is_expanded:
            self.details_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
            self.expand_btn.configure(text="▲ Hide Text", text_color=Theme.PRIMARY)
        else:
            self.details_frame.grid_forget()
            self.expand_btn.configure(text="▼ Show Text", text_color=Theme.TEXT_SECONDARY)
