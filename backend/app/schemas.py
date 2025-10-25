from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr
from enum import Enum

class UserRole(str, Enum):
    admin = "admin"
    docente = "docente"
    estudiante = "estudiante"

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserBase(BaseModel):
    email: EmailStr
    nombres: str
    apellidos: str
    rol: UserRole

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: int

    class Config:
        from_attributes = True

class StudentCreate(BaseModel):
    codigo: str
    carrera: Optional[str] = None
    uses_glasses: Optional[bool] = False
    uses_cap: Optional[bool] = False
    uses_mask: Optional[bool] = False

class StudentOut(BaseModel):
    id: int
    user_id: int
    codigo: str
    carrera: Optional[str]
    uses_glasses: bool
    uses_cap: bool
    uses_mask: bool

    class Config:
        from_attributes = True

class CourseCreate(BaseModel):
    nombre: str
    codigo: str
    horario: Optional[str] = None

class CourseOut(BaseModel):
    id: int
    docente_id: int
    nombre: str
    codigo: str
    horario: Optional[str]

    class Config:
        from_attributes = True

class EnrollmentCreate(BaseModel):
    estudiante_id: int
    curso_id: int

class EnrollmentOut(BaseModel):
    id: int
    estudiante_id: int
    curso_id: int

    class Config:
        from_attributes = True

class SelfEnrollRequest(BaseModel):
    curso_id: int

class SelfEnrollByCodeRequest(BaseModel):
    codigo: str

class AttendanceState(str, Enum):
    presente = "presente"
    ausente = "ausente"

class AttendanceCreate(BaseModel):
    estudiante_id: int
    curso_id: int
    estado: AttendanceState

class AttendanceOut(BaseModel):
    id: int
    estudiante_id: int
    curso_id: int
    estado: AttendanceState
    fecha_hora: Optional[datetime] = None

    class Config:
        from_attributes = True

# --- Auth/registration request models ---
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    nombres: str
    apellidos: str
    rol: UserRole
    # Optional student fields if rol == estudiante
    codigo: Optional[str] = None
    carrera: Optional[str] = None
    uses_glasses: Optional[bool] = False
    uses_cap: Optional[bool] = False
    uses_mask: Optional[bool] = False

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
