import os
from pathlib import Path


FILE_STORAGE_ROOT = Path(os.getenv("FILE_STORAGE_ROOT", "app/uploads"))
MAX_FILE_BYTES = int(os.getenv("FILE_MAX_BYTES", str(10 * 1024 * 1024)))

# Add new storage types here. Each type gets its own folder under FILE_STORAGE_ROOT.
ALLOWED_FILE_TYPES = {
    "face", "avatar"
}
