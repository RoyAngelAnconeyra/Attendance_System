from typing import List, Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from ...deps import get_db, get_current_user, require_role
from ... import models, schemas

router = APIRouter()

@router.get("/attendance", response_model=List[schemas.AttendanceOut])
def list_attendance(
    course_id: Optional[int] = Query(default=None),
    on_date: Optional[date] = Query(default=None),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    q = db.query(models.Attendance)
    if course_id:
        q = q.filter(models.Attendance.curso_id == course_id)
    if on_date:
        # filter same day (UTC)
        start = datetime(on_date.year, on_date.month, on_date.day)
        end = datetime(on_date.year, on_date.month, on_date.day, 23, 59, 59)
        q = q.filter(models.Attendance.fecha_hora >= start, models.Attendance.fecha_hora <= end)

    # role filters
    if user.rol == models.UserRole.docente:
        q = q.join(models.Course, models.Course.id == models.Attendance.curso_id).filter(models.Course.docente_id == user.id)
    elif user.rol == models.UserRole.estudiante:
        # only own
        student = db.query(models.Student).filter(models.Student.user_id == user.id).first()
        if not student:
            return []
        q = q.filter(models.Attendance.estudiante_id == student.id)

    return q.all()

@router.post("/attendance/mark", response_model=schemas.AttendanceOut)
def mark_attendance(
    payload: schemas.AttendanceCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin)),
):
    student = db.get(models.Student, payload.estudiante_id)
    course = db.get(models.Course, payload.curso_id)
    if not student or not course:
        raise HTTPException(status_code=404, detail="Student or course not found")
    if user.rol == models.UserRole.docente and course.docente_id != user.id:
        raise HTTPException(status_code=403, detail="No autorizado para este curso")

    att = models.Attendance(
        estudiante_id=student.id,
        curso_id=course.id,
        estado=models.AttendanceState(payload.estado.value),
    )
    db.add(att)
    db.commit()
    db.refresh(att)
    return att
