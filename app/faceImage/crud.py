from sqlalchemy.orm import Session

from app.faceImage import model, schema


def get_face_image(db: Session, face_image_id: int) -> model.FaceImage | None:
    return db.get(model.FaceImage, face_image_id)


def get_face_images_by_employee(
    db: Session,
    employee_id: int,
) -> list[model.FaceImage]:
    return (
        db.query(model.FaceImage)
        .filter(model.FaceImage.employeeId == employee_id)
        .order_by(model.FaceImage.createdAt.asc(), model.FaceImage.id.asc())
        .all()
    )


def get_face_image_by_employee(db: Session, employee_id: int, exclude_id: int | None = None) -> model.FaceImage | None:
    query = db.query(model.FaceImage).filter(model.FaceImage.employeeId == employee_id)
    if exclude_id is not None:
        query = query.filter(model.FaceImage.id != exclude_id)
    return query.first()


def create_face_image(db: Session, payload: schema.FaceImageCreate) -> model.FaceImage:
    face_image = model.FaceImage(**payload.model_dump())
    db.add(face_image)
    db.commit()
    db.refresh(face_image)
    return face_image


def update_face_image(db: Session, face_image: model.FaceImage, payload: schema.FaceImageUpdate) -> model.FaceImage:
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(face_image, field, value)

    db.commit()
    db.refresh(face_image)
    return face_image


def delete_face_image(db: Session, face_image: model.FaceImage) -> None:
    db.delete(face_image)
    db.commit()
