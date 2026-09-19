from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.auth.security import hash_password
from app.employee import model, schema

def get_employee(db: Session, employee_id: int) -> model.Employee | None:
    return db.get(model.Employee, employee_id)


def get_employees(db: Session, skip: int = 0, limit: int = 100) -> list[model.Employee]:
    return (
        db.query(model.Employee)
        .filter(model.Employee.role != "admin")
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_employee(db: Session, employee: schema.EmployeeCreate) -> model.Employee:
    # Check username trùng
    existing_employee = (db.query(model.Employee).filter(model.Employee.username == employee.username).first())

    if existing_employee:
        raise HTTPException(status_code=400, detail="Username already exists")

    employee_data = employee.model_dump()
    employee_data["password"] = hash_password(employee.password)
    db_employee = model.Employee(**employee_data)
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return db_employee


def update_employee(db: Session, db_employee: model.Employee, employee_update: schema.EmployeeUpdate) -> model.Employee:
    for field, value in employee_update.model_dump(exclude_unset=True).items():
        if field == "password":
            value = hash_password(value)
        setattr(db_employee, field, value)

    db.commit()
    db.refresh(db_employee)
    return db_employee


def delete_employee(db: Session, db_employee: model.Employee) -> None:
    db.delete(db_employee)
    db.commit()
