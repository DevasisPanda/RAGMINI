"""
File Manager for handling local PDF files and Google Drive PDF downloads.
"""

import os
import re
import shutil
import tempfile
import atexit
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import requests

logger = logging.getLogger(__name__)


class FileManager:
    """Manages local PDF selection, Google Drive PDF downloads, and temp directory cleanup."""

    def __init__(self):
        self._selected_files: List[Path] = []
        self._temp_dirs: List[str] = []
        atexit.register(self.cleanup_temp_files)

    def get_selected_files(self) -> List[Path]:
        """Return the current list of selected PDF file paths."""
        return list(self._selected_files)

    def add_files(self, paths: List[Path]) -> Tuple[int, List[str]]:
        """Add local PDF files to selection, skipping duplicates and non-PDFs.
        
        Returns:
            Tuple of (added_count, list_of_warning_messages)
        """
        added = 0
        warnings = []

        for path in paths:
            path_obj = Path(path).resolve()
            if not path_obj.exists():
                warnings.append(f"File not found: {path_obj.name}")
                continue

            if path_obj.suffix.lower() != ".pdf":
                warnings.append(f"Skipped non-PDF file: {path_obj.name}")
                continue

            if self._is_duplicate(path_obj):
                warnings.append(f"Skipped duplicate PDF: {path_obj.name}")
                continue

            self._selected_files.append(path_obj)
            added += 1

        return added, warnings

    def clear_files(self) -> None:
        """Clear all selected PDF files."""
        self._selected_files.clear()

    def remove_file(self, path: Path) -> None:
        """Remove a specific file from the selection."""
        self._selected_files = [f for f in self._selected_files if f != Path(path).resolve()]

    def download_google_drive_pdf(self, url: str) -> Tuple[Optional[Path], str]:
        """Download a PDF file from a Google Drive sharing link.
        
        Returns:
            Tuple of (downloaded_file_path, status_message)
        """
        file_id = self._extract_drive_file_id(url)
        if not file_id:
            return None, "Invalid Google Drive link format. Direct file links are supported."

        temp_dir = tempfile.mkdtemp(prefix="tinyrag_drive_")
        self._temp_dirs.append(temp_dir)

        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()

        try:
            response = session.get(download_url, stream=True, timeout=30)
            
            # Handle large file confirmation page from Google Drive
            confirm_token = self._get_confirm_token(response)
            if confirm_token:
                params = {'id': file_id, 'confirm': confirm_token}
                response = session.get(download_url, params=params, stream=True, timeout=30)

            response.raise_for_status()

            # Check if response is HTML error page instead of PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type and 'google.com' in response.text[:500]:
                return None, "Could not download file. Ensure file sharing is set to 'Anyone with the link'."

            # Determine filename from Content-Disposition header or default
            filename = self._extract_filename(response) or f"drive_{file_id[:8]}.pdf"
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"

            out_path = Path(temp_dir) / filename
            with open(out_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)

            if out_path.stat().st_size < 100:
                out_path.unlink(missing_ok=True)
                return None, "Downloaded file is empty or invalid. Verify Drive link permission."

            logger.info(f"[OK] Downloaded Google Drive PDF: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
            return out_path, f"Successfully downloaded '{out_path.name}'"

        except Exception as e:
            logger.error(f"Google Drive download failed: {e}")
            return None, f"Failed to download: {e}"

    def _is_duplicate(self, new_path: Path) -> bool:
        """Check if a file with same name and size is already selected."""
        for existing in self._selected_files:
            try:
                if existing.name == new_path.name and existing.stat().st_size == new_path.stat().st_size:
                    return True
            except OSError:
                continue
        return False

    @staticmethod
    def _extract_drive_file_id(url: str) -> Optional[str]:
        """Extract Google Drive file ID from standard sharing URLs."""
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/d/([a-zA-Z0-9_-]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _get_confirm_token(response: requests.Response) -> Optional[str]:
        """Extract confirm token for large Google Drive downloads."""
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _extract_filename(response: requests.Response) -> Optional[str]:
        """Extract filename from Content-Disposition header."""
        disp = response.headers.get('Content-Disposition')
        if disp:
            match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', disp)
            if match:
                return match.group(1)
        return None

    def cleanup_temp_files(self) -> None:
        """Remove all temporary directories created for downloads."""
        for temp_dir in self._temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
        self._temp_dirs.clear()
