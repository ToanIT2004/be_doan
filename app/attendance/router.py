from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.attendance import crud, schema
from app.auth.security import get_current_employee, require_admin
from app.database import get_db
from app.employee.crud import get_employee
from app.employee.model import Employee


router = APIRouter(prefix="/api/v1", tags=["attendances"])


@router.get("/employee/attendance", response_model=list[schema.AttendanceRead])
def get_employee_attendances(db: Session = Depends(get_db), current_employee: Employee = Depends(get_current_employee)):
    return crud.get_employee_attendances(db, current_employee.id)


@router.get("/admin/attendance/{employeeId}", response_model=list[schema.AttendanceRead])
def get_admin_employee_attendances(employeeId: int = Path(gt=0), db: Session = Depends(get_db), _admin: Employee = Depends(require_admin)):
    if get_employee(db, employeeId) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found",
        )

    return crud.get_employee_attendances(db, employeeId)
