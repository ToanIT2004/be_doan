import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.auth.router import router as auth_router
from app.attendance.router import router as attendance_router
from app.detection.router import router as detection_router
from app.employee.router import router as employee_router
from app.faceImage.router import router as face_image_router
from app.file.router import router as file_router
from app.leaveRequest.router import router as leave_request_router
from app.recognition.router import router as recognition_router


app = FastAPI(title="Doan API", version="1.0.0")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(attendance_router)
app.include_router(employee_router)
app.include_router(face_image_router)
app.include_router(file_router)
app.include_router(leave_request_router)
app.include_router(detection_router)
app.include_router(recognition_router)
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Doan API is running"}
