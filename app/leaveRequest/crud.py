from sqlalchemy.orm import Session

from app.leaveRequest import model, schema


def get_leave_request(db: Session, leave_request_id: int) -> model.LeaveRequest | None:
    return db.get(model.LeaveRequest, leave_request_id)


def get_employee_leave_request(
    db: Session,
    leave_request_id: int,
    employee_id: int,
) -> model.LeaveRequest | None:
    return (
        db.query(model.LeaveRequest)
        .filter(
            model.LeaveRequest.id == leave_request_id,
            model.LeaveRequest.employeeId == employee_id,
        )
        .first()
    )


def get_employee_leave_requests(db: Session, employee_id: int) -> list[model.LeaveRequest]:
    return (
        db.query(model.LeaveRequest)
        .filter(model.LeaveRequest.employeeId == employee_id)
        .order_by(model.LeaveRequest.id.desc())
        .all()
    )


def get_all_leave_requests(db: Session) -> list[model.LeaveRequest]:
    return db.query(model.LeaveRequest).order_by(model.LeaveRequest.id.desc()).all()


def create_leave_request(
    db: Session,
    employee_id: int,
    payload: schema.LeaveRequestCreate,
) -> model.LeaveRequest:
    leave_request = model.LeaveRequest(
        **payload.model_dump(),
        employeeId=employee_id,
        status="pending",
    )
    db.add(leave_request)
    db.commit()
    db.refresh(leave_request)
    return leave_request


def update_leave_request(
    db: Session,
    leave_request: model.LeaveRequest,
    payload: schema.LeaveRequestUpdate,
) -> model.LeaveRequest:
    for field, value in payload.model_dump(exclude={"id"}, exclude_none=True).items():
        setattr(leave_request, field, value)
    db.commit()
    db.refresh(leave_request)
    return leave_request


def delete_leave_request(db: Session, leave_request: model.LeaveRequest) -> None:
    db.delete(leave_request)
    db.commit()


def review_leave_request(
    db: Session,
    leave_request: model.LeaveRequest,
    payload: schema.LeaveRequestReview,
) -> model.LeaveRequest:
    leave_request.status = payload.status
    leave_request.rejectReason = payload.rejectReason
    db.commit()
    db.refresh(leave_request)
    return leave_request
