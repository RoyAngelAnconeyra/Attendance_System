from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.routes import auth, courses, students, attendance, face
from .db.session import Base, engine

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.API_PREFIX, tags=["auth"])
app.include_router(courses.router, prefix=settings.API_PREFIX, tags=["courses"])
app.include_router(students.router, prefix=settings.API_PREFIX, tags=["students"])
app.include_router(attendance.router, prefix=settings.API_PREFIX, tags=["attendance"])
app.include_router(face.router, prefix=settings.API_PREFIX, tags=["face"])


@app.get("/")
def read_root():
    return {"app": settings.APP_NAME, "status": "ok"}


@app.on_event("startup")
def on_startup():
    # Create DB tables if not exist (use Alembic later for migrations)
    Base.metadata.create_all(bind=engine)
