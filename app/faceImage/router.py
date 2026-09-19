from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.security import require_admin
from app.database import get_db
from app.employee.crud import get_employee
from app.faceImage import crud, schema


router = APIRouter(
    prefix="/api/v1/admin/employee/face-images",
    tags=["faceImages"],
    dependencies=[Depends(require_admin)],
)


def _require_employee(db: Session, employee_id: int) -> None:
    if get_employee(db, employee_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Employee not found")


def _require_face_image(db: Session, face_image_id: int):
    face_image = crud.get_face_image(db, face_image_id)
    if face_image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face image not found",
        )
    return face_image


def _ensure_employee_has_no_face_image(
    db: Session,
    employee_id: int,
    exclude_id: int | None = None,
) -> None:
    if crud.get_face_image_by_employee(db, employee_id, exclude_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Employee already has a face image record",
        )


def _conflict_from_integrity_error(db: Session, exc: IntegrityError) -> NoReturn:
    db.rollback()
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Employee already has a face image record",
    ) from exc


@router.get("/{employeeId}", response_model=list[schema.FaceImageRead])
def get_face_images_by_employee(
    employeeId: int,
    db: Session = Depends(get_db),
):
    _require_employee(db, employeeId)
    return crud.get_face_images_by_employee(db, employeeId)


@router.post("", response_model=schema.FaceImageRead, status_code=status.HTTP_201_CREATED)
def create_face_image(payload: schema.FaceImageCreate, db: Session = Depends(get_db)):
    _require_employee(db, payload.employeeId)
    _ensure_employee_has_no_face_image(db, payload.employeeId)
    try:
        return crud.create_face_image(db, payload)
    except IntegrityError as exc:
        _conflict_from_integrity_error(db, exc)


@router.put("/{id}", response_model=schema.FaceImageRead)
def update_face_image(id: int, payload: schema.FaceImageUpdate, db: Session = Depends(get_db)):
    face_image = _require_face_image(db, id)
    employee_id = payload.employeeId or face_image.employeeId
    if payload.employeeId is not None:
        _require_employee(db, payload.employeeId)
        _ensure_employee_has_no_face_image(db, employee_id, exclude_id=id)

    try:
        return crud.update_face_image(db, face_image, payload)
    except IntegrityError as exc:
        _conflict_from_integrity_error(db, exc)


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_face_image(id: int, db: Session = Depends(get_db)) -> None:
    face_image = _require_face_image(db, id)
    crud.delete_face_image(db, face_image)
