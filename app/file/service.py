import secrets
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

from app.file.config import ALLOWED_FILE_TYPES, FILE_STORAGE_ROOT, MAX_FILE_BYTES


class InvalidFileError(ValueError):
    """Raised when the upload type, name, or content is invalid."""


class StoredFileNotFoundError(FileNotFoundError):
    """Raised when a requested stored file does not exist."""


def validate_file_type(file_type: str) -> str:
    normalized_type = file_type.strip().lower()
    if normalized_type not in ALLOWED_FILE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_FILE_TYPES))
        raise InvalidFileError(f"Unsupported file type. Allowed types: {allowed}")
    return normalized_type


def validate_file_name(file_name: str) -> str:
    if not file_name or Path(file_name).name != file_name or file_name in {".", ".."}:
        raise InvalidFileError("Invalid file name")
    return file_name


def get_file_path(file_type: str, file_name: str) -> Path:
    normalized_type = validate_file_type(file_type)
    safe_file_name = validate_file_name(file_name)
    return FILE_STORAGE_ROOT / normalized_type / safe_file_name


def _new_file_name(extension: str, destination_dir: Path) -> str:
    # Readable by year/month/day and unique even for uploads in the same millisecond.
    while True:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        candidate = f"{timestamp}_{secrets.token_hex(2)}{extension}"
        if not (destination_dir / candidate).exists():
            return candidate


async def save_file(upload: UploadFile, file_type: str) -> str:
    normalized_type = validate_file_type(file_type)
    extension = Path(upload.filename or "").suffix.lower()
    if not extension or len(extension) > 16:
        raise InvalidFileError("The uploaded file must have a valid extension")

    destination_dir = FILE_STORAGE_ROOT / normalized_type
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_name = _new_file_name(extension, destination_dir)
    destination = destination_dir / file_name
    total_bytes = 0

    try:
        with destination.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_BYTES:
                    raise InvalidFileError(
                        f"File is too large; maximum is {MAX_FILE_BYTES} bytes"
                    )
                output.write(chunk)

        if total_bytes == 0:
            raise InvalidFileError("File is empty")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    # Return the public URL path exposed by the StaticFiles mount in app.main.
    # This value can be stored directly as FaceImage.imagePath.
    return f"/uploads/{normalized_type}/{file_name}"


def find_file(file_type: str, file_name: str) -> Path:
    file_path = get_file_path(file_type, file_name)
    if not file_path.is_file():
        raise StoredFileNotFoundError("File not found")
    return file_path


def delete_file(file_type: str, file_name: str) -> str:
    file_path = find_file(file_type, file_name)
    file_path.unlink()
    return file_path.name
