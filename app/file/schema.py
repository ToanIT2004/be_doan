from pydantic import BaseModel


class FileUploadResponse(BaseModel):
    fileName: str


class FileDeleteResponse(BaseModel):
    fileName: str
    message: str
