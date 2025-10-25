from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ...deps import get_db
from ...core.security import get_password_hash, verify_password, create_access_token
from ... import models
from ... import schemas

router = APIRouter()

@router.post("/register", response_model=schemas.UserOut)
def register(payload: schemas.RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email ya registrado")

    if payload.rol == schemas.UserRole.estudiante and not payload.codigo:
        raise HTTPException(status_code=400, detail="El CUI (codigo) es requerido para estudiantes")

    user = models.User(
        email=payload.email,
        password_hash=get_password_hash(payload.password),
        nombres=payload.nombres,
        apellidos=payload.apellidos,
        rol=models.UserRole(payload.rol.value),
    )
    db.add(user)
    db.flush()

    if payload.rol == schemas.UserRole.estudiante:
        student = models.Student(
            user_id=user.id,
            codigo=payload.codigo,  # CUI
            carrera=payload.carrera,
            uses_glasses=1 if payload.uses_glasses else 0,
            uses_cap=1 if payload.uses_cap else 0,
            uses_mask=1 if payload.uses_mask else 0,
        )
        db.add(student)

    db.commit()
    db.refresh(user)
    return user

@router.post("/login", response_model=schemas.Token)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales inválidas")

    token = create_access_token({"sub": str(user.id), "role": user.rol.value})
    return {"access_token": token, "token_type": "bearer"}
