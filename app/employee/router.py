from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.security import require_admin
from app.database import get_db
from app.employee import crud, schema

router = APIRouter(prefix="/api/v1", tags=["employees"])

@router.get("/admin/employee", response_model=list[schema.EmployeeRead], dependencies=[Depends(require_admin)])
def get_all_employees(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_employees(db, skip=skip, limit=limit)


@router.get("/admin/employee/{employee_id}", response_model=schema.EmployeeRead, dependencies=[Depends(require_admin)])
def get_employee_by_id(employee_id: int, db: Session = Depends(get_db)):
    db_employee = crud.get_employee(db, employee_id)
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return db_employee


@router.post("/admin/employee", response_model=schema.EmployeeRead, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
def create_employee(employee: schema.EmployeeCreate, db: Session = Depends(get_db)):
    return crud.create_employee(db, employee)


@router.put("/admin/employee/{employee_id}", response_model=schema.EmployeeRead, dependencies=[Depends(require_admin)])
def update_employee(employee_id: int, employee_update: schema.EmployeeUpdate, db: Session = Depends(get_db)):
    db_employee = crud.get_employee(db, employee_id)
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return crud.update_employee(db, db_employee, employee_update)


@router.delete("/admin/employee/{employee_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)])
def delete_employee(employee_id: int, db: Session = Depends(get_db)):
    db_employee = crud.get_employee(db, employee_id)
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    crud.delete_employee(db, db_employee)
