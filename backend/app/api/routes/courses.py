from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ...deps import get_db, get_current_user, require_role
from ... import models, schemas

router = APIRouter()

@router.get("/courses", response_model=List[schemas.CourseOut])
def list_courses(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    if user.rol == models.UserRole.docente:
        return db.query(models.Course).filter(models.Course.docente_id == user.id).all()
    elif user.rol == models.UserRole.estudiante:
        return (
            db.query(models.Course)
            .join(models.Enrollment, models.Enrollment.curso_id == models.Course.id)
            .join(models.Student, models.Student.id == models.Enrollment.estudiante_id)
            .filter(models.Student.user_id == user.id)
            .all()
        )
    else:
        return db.query(models.Course).all()

@router.post("/courses", response_model=schemas.CourseOut)
def create_course(
    payload: schemas.CourseCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin)),
):
    if db.query(models.Course).filter(models.Course.codigo == payload.codigo).first():
        raise HTTPException(status_code=400, detail="Código de curso ya existe")

    docente_id = user.id if user.rol == models.UserRole.docente else user.id  # admin can also create for self
    course = models.Course(
        docente_id=docente_id,
        nombre=payload.nombre,
        codigo=payload.codigo,
        horario=payload.horario,
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


@router.get("/courses/browse", response_model=List[schemas.CourseOut])
def browse_courses(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    # For estudiantes: show all courses so they can self-enroll. For others: same as list all.
    return db.query(models.Course).all()
