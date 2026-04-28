"""Bundle generated artifacts into an in-memory zip."""
from __future__ import annotations

import io
import zipfile


def build_zip(files: dict[str, str], root: str = "easy-vllm-output") -> bytes:
    """Create a zip with each file under ``root/`` and return the bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files.items():
            arcname = f"{root}/{filename}"
            zf.writestr(arcname, content)
    return buffer.getvalue()
