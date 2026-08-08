"""
ChatPanel for question submission, answer rendering, structured citations, and copy/clear controls.
"""

import customtkinter as ctk
from typing import Callable, Optional, Dict, List
from pathlib import Path

from ..theme import Theme
from ..widgets.animated_button import AnimatedButton
from ..widgets.citation_card import CitationCard
from Rag.core.models import QueryResult, Citation


class ChatPanel(ctk.CTkFrame):
    """Panel containing the chat output area, question input, and action controls."""

    def __init__(
        self,
        master: any,
        on_ask_submitted: Callable[[str], None],
        on_clear_chat: Callable[[], None],
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
        self.on_ask_submitted = on_ask_submitted
        self.on_clear_chat = on_clear_chat
        self.current_answer: str = ""

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
            text="Document Question Answering",
            font=Theme.FONT_TITLE,
            text_color=Theme.TEXT_PRIMARY
        )
        lbl_title.grid(row=0, column=0, sticky="w")

        btn_copy = AnimatedButton(
            header_frame, text="Copy Answer", command=self._copy_answer, variant="ghost", icon="📋", width=100, height=26
        )
        btn_copy.grid(row=0, column=1, padx=(0, 6), sticky="e")

        btn_clear = AnimatedButton(
            header_frame, text="Clear Chat", command=self._handle_clear, variant="ghost", icon="🗑️", width=90, height=26
        )
        btn_clear.grid(row=0, column=2, sticky="e")

        # Scrollable Output Container
        self.output_scroll = ctk.CTkScrollableFrame(
            self,
            fg_color=Theme.BG_DARK,
            corner_radius=Theme.RADIUS_SM
        )
        self.output_scroll.grid(row=1, column=0, sticky="nsew", padx=Theme.PAD_MD, pady=Theme.PAD_XS)
        self.output_scroll.grid_columnconfigure(0, weight=1)

        # Show empty placeholder initial state
        self._show_empty_placeholder()

        # Input Row
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=2, column=0, sticky="ew", padx=Theme.PAD_MD, pady=Theme.PAD_MD)
        input_frame.grid_columnconfigure(0, weight=1)

        self.query_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter your question here... (e.g., What guidelines were laid down in D.K. Basu?)",
            font=Theme.FONT_BODY,
            height=40,
            border_color=Theme.BORDER_LIGHT,
            fg_color=Theme.SURFACE_VARIANT
        )
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=(0, Theme.PAD_SM))
        self.query_entry.bind("<Return>", lambda e: self._submit_query())

        self.btn_ask = AnimatedButton(
            input_frame, text="Ask Question", command=self._submit_query, variant="primary", icon="💬", width=120, height=40
        )
        self.btn_ask.grid(row=0, column=1, sticky="e")

    def _show_empty_placeholder(self) -> None:
        """Display helpful guidance placeholder when no query has been asked."""
        for child in self.output_scroll.winfo_children():
            child.destroy()

        placeholder = ctk.CTkLabel(
            self.output_scroll,
            text="💬 Ask questions about your indexed PDF documents\n\nExample: 'What procedural safeguards exist for arrested persons?'",
            font=Theme.FONT_BODY,
            text_color=Theme.TEXT_SECONDARY,
            justify="center"
        )
        placeholder.pack(expand=True, pady=60)

    def _submit_query(self) -> None:
        query = self.query_entry.get().strip()
        if query and self.on_ask_submitted:
            self.on_ask_submitted(query)

    def display_query_result(self, result: QueryResult) -> None:
        """Render a QueryResult with formatted question, answer text, and citation cards."""
        self.current_answer = result.answer

        # Clear scrollable container
        for child in self.output_scroll.winfo_children():
            child.destroy()

        # Question Banner
        q_frame = ctk.CTkFrame(self.output_scroll, fg_color=Theme.SURFACE_VARIANT, corner_radius=Theme.RADIUS_SM)
        q_frame.pack(fill="x", padx=6, pady=(6, 10))

        q_lbl = ctk.CTkLabel(
            q_frame,
            text=f"Q: {result.question}",
            font=Theme.FONT_SUBTITLE,
            text_color=Theme.PRIMARY,
            anchor="w",
            justify="left",
            wraplength=650
        )
        q_lbl.pack(fill="x", padx=12, pady=10)

        # Answer Box
        a_frame = ctk.CTkFrame(self.output_scroll, fg_color=Theme.SURFACE, corner_radius=Theme.RADIUS_SM)
        a_frame.pack(fill="x", padx=6, pady=(0, 10))

        a_title = ctk.CTkLabel(
            a_frame,
            text="Answer:",
            font=Theme.FONT_SUBTITLE,
            text_color=Theme.TEXT_PRIMARY,
            anchor="w"
        )
        a_title.pack(fill="x", padx=12, pady=(10, 4))

        ans_color = Theme.TEXT_PRIMARY if result.is_answerable else Theme.WARNING

        a_text = ctk.CTkLabel(
            a_frame,
            text=result.answer,
            font=Theme.FONT_BODY,
            text_color=ans_color,
            anchor="w",
            justify="left",
            wraplength=650
        )
        a_text.pack(fill="x", padx=12, pady=(0, 12))

        # Citations Section
        if result.citations:
            sources_lbl = ctk.CTkLabel(
                self.output_scroll,
                text="Sources & Citations:",
                font=Theme.FONT_SUBTITLE,
                text_color=Theme.TEXT_PRIMARY,
                anchor="w"
            )
            sources_lbl.pack(fill="x", padx=6, pady=(6, 4))

            # Group citations by document_name
            doc_groups: Dict[str, Dict[str, Any]] = {}
            for c in result.citations:
                doc_name = c.document_name
                if doc_name not in doc_groups:
                    doc_groups[doc_name] = {"pages": set(), "citations": []}
                doc_groups[doc_name]["pages"].add(c.page_number)
                doc_groups[doc_name]["citations"].append(c)

            for doc_name, data in doc_groups.items():
                card = CitationCard(
                    self.output_scroll,
                    document_name=doc_name,
                    pages=list(data["pages"]),
                    citations=data["citations"]
                )
                card.pack(fill="x", padx=6, pady=4)

    def _copy_answer(self) -> None:
        if self.current_answer:
            self.clipboard_clear()
            self.clipboard_append(self.current_answer)

    def _handle_clear(self) -> None:
        self.current_answer = ""
        self.query_entry.delete(0, "end")
        self._show_empty_placeholder()
        if self.on_clear_chat:
            self.on_clear_chat()

    def set_thinking_state(self, thinking: bool) -> None:
        """Toggle controls disabled state during question answering."""
        if thinking:
            self.btn_ask.set_loading(True, "Thinking...")
            self.query_entry.configure(state="disabled")
        else:
            self.btn_ask.set_loading(False)
            self.query_entry.configure(state="normal")
