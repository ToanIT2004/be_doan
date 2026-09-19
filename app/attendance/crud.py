import os
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.attendance.model import Attendance


def _app_timezone() -> tzinfo:
    timezone_name = os.getenv("APP_TIMEZONE", "Asia/Ho_Chi_Minh")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        if timezone_name in {"Asia/Ho_Chi_Minh", "Asia/Saigon"}:
            # Vietnam does not observe daylight saving time. This fallback keeps
            # Windows deployments working even when the tzdata package is absent.
            return timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")
        raise RuntimeError(f"Invalid APP_TIMEZONE: {timezone_name}") from exc


def _local_scan_time(scanned_at: datetime | None = None) -> datetime:
    timezone = _app_timezone()
    if scanned_at is None:
        return datetime.now(timezone)
    if scanned_at.tzinfo is None:
        return scanned_at.replace(tzinfo=timezone)
    return scanned_at.astimezone(timezone)


def get_daily_attendance(
    db: Session,
    employee_id: int,
    attendance_date,
) -> Attendance | None:
    return (
        db.query(Attendance)
        .filter(
            Attendance.employeeId == employee_id,
            Attendance.attendanceDate == attendance_date,
        )
        .first()
    )


def get_employee_attendances(db: Session, employee_id: int) -> list[Attendance]:
    return (
        db.query(Attendance)
        .filter(Attendance.employeeId == employee_id)
        .order_by(Attendance.attendanceDate.desc(), Attendance.id.desc())
        .all()
    )


def mark_attendance(
    db: Session,
    employee_id: int,
    scanned_at: datetime | None = None,
) -> Attendance:
    local_time = _local_scan_time(scanned_at)
    attendance_date = local_time.date()
    timestamp = local_time.isoformat(timespec="seconds")
    attendance = get_daily_attendance(db, employee_id, attendance_date)

    if attendance is not None:
        attendance.checkOut = timestamp
        db.commit()
        db.refresh(attendance)
        return attendance

    attendance = Attendance(
        employeeId=employee_id,
        attendanceDate=attendance_date,
        checkIn=timestamp,
        checkOut=None,
    )
    db.add(attendance)
    try:
        db.commit()
    except IntegrityError:
        # A concurrent first scan may have created today's row already.
        db.rollback()
        attendance = get_daily_attendance(db, employee_id, attendance_date)
        if attendance is None:
            raise
        attendance.checkOut = timestamp
        db.commit()

    db.refresh(attendance)
    return attendance
