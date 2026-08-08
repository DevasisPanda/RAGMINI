"""
StatusBar panel showing application state, provider badge, and document counts.
"""

import customtkinter as ctk
from ..theme import Theme


class StatusBar(ctk.CTkFrame):
    """Top status bar container."""

    def __init__(self, master: any, **kwargs):
        super().__init__(
            master,
            fg_color=Theme.SURFACE,
            border_color=Theme.BORDER,
            border_width=1,
            height=36,
            corner_radius=0,
            **kwargs
        )
        self.grid_columnconfigure(1, weight=1)

        self._build_ui()

    def _build_ui(self) -> None:
        # Left status indicator dot + label
        self.status_dot = ctk.CTkLabel(
            self,
            text="●",
            font=("Segoe UI", 14),
            text_color=Theme.SUCCESS
        )
        self.status_dot.grid(row=0, column=0, padx=(12, 4), pady=4, sticky="w")

        self.status_text = ctk.CTkLabel(
            self,
            text="Ready",
            font=Theme.FONT_BODY,
            text_color=Theme.TEXT_PRIMARY
        )
        self.status_text.grid(row=0, column=1, padx=4, pady=4, sticky="w")

        # Provider Badge
        self.provider_badge = ctk.CTkLabel(
            self,
            text="Provider: OpenRouter (gemma-2-9b)",
            font=Theme.FONT_SMALL,
            text_color=Theme.TEXT_SECONDARY,
            fg_color=Theme.SURFACE_VARIANT,
            corner_radius=Theme.RADIUS_SM,
            padx=10,
            pady=3
        )
        self.provider_badge.grid(row=0, column=2, padx=8, pady=4, sticky="e")

        # Document count badge
        self.doc_badge = ctk.CTkLabel(
            self,
            text="PDFs: 0 selected",
            font=Theme.FONT_SMALL,
            text_color=Theme.TEXT_SECONDARY,
            fg_color=Theme.SURFACE_VARIANT,
            corner_radius=Theme.RADIUS_SM,
            padx=10,
            pady=3
        )
        self.doc_badge.grid(row=0, column=3, padx=(0, 12), pady=4, sticky="e")

    def set_status(self, state: str, message: str) -> None:
        """Update status dot color and message."""
        self.status_text.configure(text=message)

        if state == "ready":
            self.status_dot.configure(text_color=Theme.SUCCESS)
        elif state == "indexing":
            self.status_dot.configure(text_color=Theme.WARNING)
        elif state == "thinking":
            self.status_dot.configure(text_color=Theme.PRIMARY)
        elif state == "error":
            self.status_dot.configure(text_color=Theme.ERROR)
        else:
            self.status_dot.configure(text_color=Theme.TEXT_SECONDARY)

    def set_provider_info(self, provider_name: str, model_name: str) -> None:
        """Update provider badge display."""
        short_model = model_name.split('/')[-1] if '/' in model_name else model_name
        self.provider_badge.configure(text=f"Provider: {provider_name} ({short_model})")

    def set_doc_count(self, count: int) -> None:
        """Update selected PDF document count badge."""
        self.doc_badge.configure(text=f"PDFs: {count} selected")
