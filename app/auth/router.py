from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import schema
from app.auth.security import create_access_token, get_current_employee, hash_password, verify_password
from app.database import get_db
from app.employee import model as employee_model


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

@router.post("/login", response_model=schema.TokenResponse)
def login(login_data: schema.LoginRequest, db: Session = Depends(get_db)):
    employee = (db.query(employee_model.Employee).filter(employee_model.Employee.username == login_data.username).first())

    if employee is None or not verify_password(login_data.password, employee.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Username or password is incorrect", headers={"WWW-Authenticate": "Bearer"})

    if employee.password == login_data.password:
        employee.password = hash_password(login_data.password)
        db.commit()
        db.refresh(employee)

    return {
        "employee": employee,
        "accessToken": create_access_token(employee),
    }


@router.get("/me", response_model=schema.CurrentEmployee)
def get_me(current_employee: employee_model.Employee = Depends(get_current_employee)):
    return current_employee
