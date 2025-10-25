from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Enum as SAEnum, UniqueConstraint
from sqlalchemy.orm import relationship
from .db.session import Base

class UserRole(str, Enum):
    admin = "admin"
    docente = "docente"
    estudiante = "estudiante"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    nombres = Column(String(255), nullable=False)
    apellidos = Column(String(255), nullable=False)
    rol = Column(SAEnum(UserRole), nullable=False, index=True)

    docente_courses = relationship("Course", back_populates="docente", cascade="all, delete-orphan")
    student = relationship("Student", back_populates="user", uselist=False)

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    codigo = Column(String(50), unique=True, index=True, nullable=False)  # CUI
    carrera = Column(String(255), nullable=True)
    uses_glasses = Column(Integer, default=0)  # 0/1 flags; keep simple
    uses_cap = Column(Integer, default=0)
    uses_mask = Column(Integer, default=0)
    # Optional: store a mean embedding as CSV string for later incremental training
    embedding = Column(String, nullable=True)

    user = relationship("User", back_populates="student")
    enrollments = relationship("Enrollment", back_populates="student", cascade="all, delete-orphan")
    attendance = relationship("Attendance", back_populates="student", cascade="all, delete-orphan")

class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    docente_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    nombre = Column(String(255), nullable=False)
    codigo = Column(String(50), unique=True, index=True, nullable=False)
    horario = Column(String(255), nullable=True)

    docente = relationship("User", back_populates="docente_courses")
    enrollments = relationship("Enrollment", back_populates="course", cascade="all, delete-orphan")
    attendance = relationship("Attendance", back_populates="course", cascade="all, delete-orphan")

class Enrollment(Base):
    __tablename__ = "enrollments"
    __table_args__ = (
        UniqueConstraint("estudiante_id", "curso_id", name="uq_enrollment_student_course"),
    )

    id = Column(Integer, primary_key=True, index=True)
    estudiante_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    curso_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)

    student = relationship("Student", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")

class AttendanceState(str, Enum):
    presente = "presente"
    ausente = "ausente"

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    estudiante_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    curso_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    fecha_hora = Column(DateTime, default=datetime.utcnow, index=True)
    estado = Column(SAEnum(AttendanceState), nullable=False)

    student = relationship("Student", back_populates="attendance")
    course = relationship("Course", back_populates="attendance")
