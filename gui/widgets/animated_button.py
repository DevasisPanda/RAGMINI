"""
AnimatedButton widget providing styled action buttons with loading states.
"""

import customtkinter as ctk
from typing import Optional, Callable
from ..theme import Theme


class AnimatedButton(ctk.CTkButton):
    """Button subclass with custom colors and loading state support."""

    def __init__(
        self,
        master: any,
        text: str,
        command: Optional[Callable[[], None]] = None,
        variant: str = "primary",
        icon: str = "",
        **kwargs
    ):
        self.normal_text = f"{icon} {text}".strip()
        self.variant = variant

        bg_color, hover_color, text_color = self._get_variant_colors(variant)

        super().__init__(
            master,
            text=self.normal_text,
            command=command,
            fg_color=bg_color,
            hover_color=hover_color,
            text_color=text_color,
            font=Theme.FONT_SUBTITLE,
            corner_radius=Theme.RADIUS_SM,
            **kwargs
        )

    def set_loading(self, loading: bool, loading_text: str = "Processing...") -> None:
        """Toggle button loading state (disabled + text change)."""
        if loading:
            self.configure(state="disabled", text=f"⏳ {loading_text}")
        else:
            self.configure(state="normal", text=self.normal_text)

    @staticmethod
    def _get_variant_colors(variant: str):
        if variant == "primary":
            return Theme.PRIMARY, Theme.PRIMARY_HOVER, Theme.TEXT_PRIMARY
        elif variant == "success":
            return Theme.SUCCESS, "#059669", Theme.TEXT_PRIMARY
        elif variant == "danger":
            return Theme.ERROR, "#DC2626", Theme.TEXT_PRIMARY
        elif variant == "secondary":
            return Theme.SURFACE_HOVER, Theme.BORDER_LIGHT, Theme.TEXT_PRIMARY
        elif variant == "ghost":
            return "transparent", Theme.SURFACE_HOVER, Theme.TEXT_SECONDARY
        else:
            return Theme.PRIMARY, Theme.PRIMARY_HOVER, Theme.TEXT_PRIMARY
