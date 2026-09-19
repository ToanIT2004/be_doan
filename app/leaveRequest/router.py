from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.security import get_current_employee, require_admin
from app.database import get_db
from app.employee.model import Employee
from app.leaveRequest import crud, schema


router = APIRouter(prefix="/api/v1", tags=["leaveRequests"])


def _require_owned_leave_request(
    db: Session,
    leave_request_id: int,
    employee_id: int,
):
    leave_request = crud.get_employee_leave_request(db, leave_request_id, employee_id)
    if leave_request is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Leave request not found",
        )
    return leave_request


def _require_pending(leave_request) -> None:
    if leave_request.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only pending leave requests can be changed",
        )


@router.post(
    "/employee/leave-request",
    response_model=schema.LeaveRequestRead,
    status_code=status.HTTP_201_CREATED,
)
def create_leave_request(
    payload: schema.LeaveRequestCreate,
    db: Session = Depends(get_db),
    current_employee: Employee = Depends(get_current_employee),
):
    return crud.create_leave_request(db, current_employee.id, payload)


@router.put(
    "/employee/leave-request",
    response_model=schema.LeaveRequestRead,
)
def update_leave_request(
    payload: schema.LeaveRequestUpdate,
    db: Session = Depends(get_db),
    current_employee: Employee = Depends(get_current_employee),
):
    leave_request = _require_owned_leave_request(db, payload.id, current_employee.id)
    _require_pending(leave_request)

    start_date = payload.startDate or leave_request.startDate
    end_date = payload.endDate or leave_request.endDate
    if end_date < start_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="endDate must be on or after startDate",
        )
    return crud.update_leave_request(db, leave_request, payload)


@router.delete(
    "/employee/leave-request/{leave_request_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_leave_request(
    leave_request_id: int,
    db: Session = Depends(get_db),
    current_employee: Employee = Depends(get_current_employee),
) -> None:
    leave_request = _require_owned_leave_request(db, leave_request_id, current_employee.id)
    _require_pending(leave_request)
    crud.delete_leave_request(db, leave_request)


@router.get(
    "/employee/leave-request/my",
    response_model=list[schema.LeaveRequestRead],
)
def get_my_leave_requests(
    db: Session = Depends(get_db),
    current_employee: Employee = Depends(get_current_employee),
):
    return crud.get_employee_leave_requests(db, current_employee.id)


@router.post(
    "/admin/leave-request/browse",
    response_model=schema.LeaveRequestRead,
)
def review_leave_request(
    payload: schema.LeaveRequestReview,
    db: Session = Depends(get_db),
    _admin: Employee = Depends(require_admin),
):
    leave_request = crud.get_leave_request(db, payload.id)
    if leave_request is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Leave request not found",
        )
    _require_pending(leave_request)
    return crud.review_leave_request(db, leave_request, payload)


@router.get(
    "/admin/leave-request",
    response_model=list[schema.LeaveRequestRead],
)
def get_all_leave_requests(
    db: Session = Depends(get_db),
    _admin: Employee = Depends(require_admin),
):
    return crud.get_all_leave_requests(db)
