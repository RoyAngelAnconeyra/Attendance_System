from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ...deps import get_db, require_role, get_current_user
from ... import models, schemas

router = APIRouter()

@router.get("/students", response_model=List[schemas.StudentOut])
def list_students(
    course_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    # Estudiante: solo sí mismo
    if user.rol == models.UserRole.estudiante:
        student = db.query(models.Student).filter(models.Student.user_id == user.id).first()
        return [student] if student else []

    # Admin sin filtro: todos los estudiantes
    if user.rol == models.UserRole.admin and course_id is None:
        return db.query(models.Student).all()

    # Para docente o admin con filtro por curso: unir una sola vez
    q = (
        db.query(models.Student)
        .join(models.Enrollment, models.Enrollment.estudiante_id == models.Student.id)
    )
    if course_id is not None:
        q = q.filter(models.Enrollment.curso_id == course_id)
    if user.rol == models.UserRole.docente:
        q = (
            q.join(models.Course, models.Course.id == models.Enrollment.curso_id)
            .filter(models.Course.docente_id == user.id)
        )
    return q.distinct(models.Student.id).all()

@router.post("/enroll", response_model=schemas.CourseOut)
def enroll_student(
    payload: schemas.EnrollmentCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin)),
):
    student = db.get(models.Student, payload.estudiante_id)
    course = db.get(models.Course, payload.curso_id)
    if not student or not course:
        raise HTTPException(status_code=404, detail="Student or course not found")
    if user.rol == models.UserRole.docente and course.docente_id != user.id:
        raise HTTPException(status_code=403, detail="No autorizado para este curso")

    exists = (
        db.query(models.Enrollment)
        .filter(models.Enrollment.estudiante_id == student.id, models.Enrollment.curso_id == course.id)
        .first()
    )
    if exists:
        return course

    enrollment = models.Enrollment(estudiante_id=student.id, curso_id=course.id)
    db.add(enrollment)
    db.commit()
    return course


@router.get("/enrollments", response_model=List[schemas.EnrollmentOut])
def list_enrollments(
    course_id: int | None = Query(default=None),
    student_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    q = db.query(models.Enrollment)
    if course_id is not None:
        q = q.filter(models.Enrollment.curso_id == course_id)
    if student_id is not None:
        q = q.filter(models.Enrollment.estudiante_id == student_id)

    if user.rol == models.UserRole.docente:
        # Only enrollments under this docente
        q = (
            q.join(models.Course, models.Course.id == models.Enrollment.curso_id)
            .filter(models.Course.docente_id == user.id)
        )
    elif user.rol == models.UserRole.estudiante:
        stu = db.query(models.Student).filter(models.Student.user_id == user.id).first()
        if not stu:
            return []
        q = q.filter(models.Enrollment.estudiante_id == stu.id)
    return q.all()


@router.get("/students/by_code", response_model=schemas.StudentOut)
def get_student_by_code(
    codigo: str = Query(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin)),
):
    stu = db.query(models.Student).filter(models.Student.codigo == codigo).first()
    if not stu:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    return stu


@router.post("/self-enroll", response_model=schemas.CourseOut)
def self_enroll(
    payload: schemas.SelfEnrollRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.estudiante)),
):
    stu = db.query(models.Student).filter(models.Student.user_id == user.id).first()
    if not stu:
        raise HTTPException(status_code=400, detail="Perfil de estudiante no encontrado")
    course = db.get(models.Course, payload.curso_id)
    if not course:
        raise HTTPException(status_code=404, detail="Curso no encontrado")
    exists = (
        db.query(models.Enrollment)
        .filter(models.Enrollment.estudiante_id == stu.id, models.Enrollment.curso_id == course.id)
        .first()
    )
    if exists:
        return course
    db.add(models.Enrollment(estudiante_id=stu.id, curso_id=course.id))
    db.commit()
    return course


@router.post("/self-enroll/by_code", response_model=schemas.CourseOut)
def self_enroll_by_code(
    payload: schemas.SelfEnrollByCodeRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.estudiante)),
):
    stu = db.query(models.Student).filter(models.Student.user_id == user.id).first()
    if not stu:
        raise HTTPException(status_code=400, detail="Perfil de estudiante no encontrado")
    course = db.query(models.Course).filter(models.Course.codigo == payload.codigo).first()
    if not course:
        raise HTTPException(status_code=404, detail="Curso no encontrado")
    exists = (
        db.query(models.Enrollment)
        .filter(models.Enrollment.estudiante_id == stu.id, models.Enrollment.curso_id == course.id)
        .first()
    )
    if exists:
        return course
    db.add(models.Enrollment(estudiante_id=stu.id, curso_id=course.id))
    db.commit()
    return course
