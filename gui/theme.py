"""
Centralized Theme & Design System tokens for TinyRAG Desktop GUI.
"""


class Theme:
    # Color Palette (Modern Slate Dark Mode)
    BG_DARK = "#0B0D17"          # Main Window Background
    SURFACE = "#141825"          # Panel Background
    SURFACE_HOVER = "#1E293B"    # Card/Item Hover Background
    SURFACE_VARIANT = "#1E293B"  # Sub-panel / Input Background

    PRIMARY = "#3B82F6"          # Primary Action Blue
    PRIMARY_HOVER = "#2563EB"    # Primary Hover Blue
    ACCENT = "#8B5CF6"           # Vibrant Purple Accent

    SUCCESS = "#10B981"          # Emerald Green Status/Success
    WARNING = "#F59E0B"          # Amber Warning/Indexing Status
    ERROR = "#EF4444"            # Rose Red Error

    TEXT_PRIMARY = "#F8FAFC"      # High contrast white
    TEXT_SECONDARY = "#94A3B8"    # Muted slate gray
    TEXT_MUTED = "#64748B"        # Subtle log timestamp gray

    BORDER = "#1E293B"           # Subtle container border
    BORDER_LIGHT = "#334155"     # Active input/card border

    # Log Level Colors
    LOG_INFO = "#60A5FA"
    LOG_OK = "#34D399"
    LOG_WARN = "#FBBF24"
    LOG_ERROR = "#F87171"

    # Typography
    FONT_FAMILY = "Segoe UI"
    FONT_MONO = "Consolas"

    FONT_TITLE = ("Segoe UI", 18, "bold")
    FONT_SUBTITLE = ("Segoe UI", 13, "bold")
    FONT_BODY = ("Segoe UI", 12)
    FONT_SMALL = ("Segoe UI", 10)
    FONT_CODE = ("Consolas", 11)

    # Geometry & Padding
    PAD_XS = 4
    PAD_SM = 8
    PAD_MD = 12
    PAD_LG = 16

    RADIUS_SM = 6
    RADIUS_MD = 10
    RADIUS_LG = 14
