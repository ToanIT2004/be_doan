from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.file import schema
from app.file.service import (InvalidFileError, StoredFileNotFoundError, delete_file, find_file, save_file)


router = APIRouter(prefix="/api/v1/file/uploads", tags=["files"])


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, StoredFileNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/upload", response_model=schema.FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    image: Annotated[UploadFile, File(description="File to upload")],
    type: Annotated[str, Form(description="Storage type configured by the server")],
) -> schema.FileUploadResponse:
    try:
        file_name = await save_file(image, type)
        return schema.FileUploadResponse(fileName=file_name)
    except InvalidFileError as exc:
        raise _http_error(exc) from exc


@router.delete("/delete/{type}/{file_name}", response_model=schema.FileDeleteResponse)
def remove_file(type: str, file_name: str) -> schema.FileDeleteResponse:
    try:
        deleted_name = delete_file(type, file_name)
        return schema.FileDeleteResponse(
            fileName=deleted_name,
            message="File deleted successfully",
        )
    except (InvalidFileError, StoredFileNotFoundError) as exc:
        raise _http_error(exc) from exc


@router.get("/{type}/{file_name}", response_class=FileResponse)
def get_file(type: str, file_name: str) -> FileResponse:
    try:
        file_path = find_file(type, file_name)
        return FileResponse(path=file_path)
    except (InvalidFileError, StoredFileNotFoundError) as exc:
        raise _http_error(exc) from exc
