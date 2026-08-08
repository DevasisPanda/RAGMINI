"""
TinyRAG Desktop Application — GUI Entry Point.

Usage: python app.py
"""

import sys
import logging

# Ensure UTF-8 stdout encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import Settings
from controllers.backend_controller import BackendController
from gui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Launch the TinyRAG Desktop GUI application."""
    settings = Settings()

    if not settings.openrouter_api_key or settings.openrouter_api_key == "your-openrouter-api-key-here":
        logger.warning(
            "OPENROUTER_API_KEY is not set in .env or environment! "
            "Please set your key in .env file before running."
        )

    # Initialize controller and GUI
    controller = BackendController(settings)
    app = MainWindow(controller)

    logger.info("Starting TinyRAG Desktop GUI application...")
    app.mainloop()


if __name__ == "__main__":
    main()
