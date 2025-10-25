# Attendance System – Face Recognition (FastAPI + React)

Sistema de asistencia con reconocimiento facial usando MTCNN (detección), FaceNet (embeddings) y SVM (clasificación). Backend en FastAPI (Python) y frontend en React (Vite + TypeScript + MUI).

## Arquitectura

- **Backend (`backend/`)**
  - FastAPI + SQLAlchemy + PostgreSQL
  - Endpoints principales:
    - `POST /api/capture/{cui}`: captura por webcam y guarda recortes en `dataset/{CUI}/{categoria}/` (MTCNN)
    - `POST /api/train`: genera embeddings con `keras-facenet` y entrena SVM global (guarda `models/face_svm.pkl` y `label_encoder.pkl`)
    - `GET /api/model/status`: estado del modelo (clases, mtime, threshold)
    - `GET /api/recognize/start`: reconocimiento (demo) por ventana de tiempo y marcado de asistencia (presente/ausente) por curso
    - `WS /api/recognize/ws`: streaming opcional de predicciones en vivo (no marca)
    - `GET /api/attendance`: consulta de asistencia por curso/fecha
    - `POST /api/self-enroll` y `/api/self-enroll/by_code`: auto-matrícula de estudiantes
  - Config: `backend/app/core/config.py` (variables vía `backend/.env`)
  - Modelos ORM: `backend/app/models.py`

- **Frontend (`frontend/`)**
  - Vite + React + TypeScript + MUI
  - Páginas principales:
    - `ProfessorDashboard.tsx`: cursos, entrenar modelo, conectar WS, iniciar demo (marca asistencia), lista de estudiantes (verde: presente con hora / rojo: ausente)
    - `StudentDashboard.tsx`: unirse a cursos, capturar fotos (webcam) con categoría (`normal/gorro/lentes/mascarilla`), ver asistencia propia
  - Cliente API: `frontend/src/api/client.ts` (Axios con token JWT)

## Requisitos

- Python 3.10+
- Node.js 18+
- PostgreSQL 13+
- (Opcional) `tzdata` para zona horaria "America/Lima"

## Instalación

1) Backend

```bash
# En la carpeta backend/
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: backend\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # Si no existe, instala manual: fastapi uvicorn[standard] sqlalchemy psycopg2-binary pydantic-settings python-multipart mtcnn keras-facenet scikit-learn opencv-python tzdata

# Variables de entorno (backend/.env)
# Edita credenciales de DB y parámetros
copy .env.example .env  # si existe; o crea backend/.env con:
```

Ejemplo de `backend/.env`:

```env
ENV=dev
API_PREFIX=/api
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=attendance_db
DB_USER=root
DB_PASSWORD=root
JWT_SECRET=CAMBIA_ESTE_VALOR
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=120
CAMERA_INDEX=0
CONFIDENCE_THRESHOLD=0.80
```

Inicia backend (terminal 1):

```bash
uvicorn app.main:app --reload --app-dir backend --host 0.0.0.0 --port 8000
```

2) Frontend (terminal 2)

```bash
# En la carpeta frontend/
npm install  # o pnpm/yarn

# Crea frontend/.env
# VITE_API_URL debe apuntar al backend (incluye /api)
# Ejemplo:
echo VITE_API_URL=http://localhost:8000/api > .env

npm run dev
```

### Ejecutar backend y frontend por separado

- **Terminal 1 (backend)**
  ```bash
  # desde la raíz del repo
  uvicorn app.main:app --reload --app-dir backend --host 0.0.0.0 --port 8000
  ```
- **Terminal 2 (frontend)**
  ```bash
  # desde frontend/
  npm run dev
  ```
- Asegúrate de que `frontend/.env` tenga `VITE_API_URL=http://localhost:8000/api` y que CORS permita tu origen (ya configurado en `backend/app/main.py`).

### Crear la base de datos (PostgreSQL)

Puedes crear la BD con SQL puro o con utilidades `createdb/psql`.

- **SQL (desde psql)**
  ```sql
  CREATE DATABASE attendance_db;
  -- 1) Tabla de usuarios
  -- Roles: admin | docente | estudiante (usamos CHECK para emular Enum)
  CREATE TABLE users (
    id               SERIAL PRIMARY KEY,
    email            VARCHAR(255) NOT NULL UNIQUE,
    password_hash    VARCHAR(255) NOT NULL,
    nombres          VARCHAR(255) NOT NULL,
    apellidos        VARCHAR(255) NOT NULL,
    rol              VARCHAR(20)  NOT NULL CHECK (rol IN ('admin','docente','estudiante'))
  );

  CREATE INDEX idx_users_email ON users(email);

  -- 2) Tabla de estudiantes (1:1 con users; un usuario solo puede tener un student)
  CREATE TABLE students (
    id             SERIAL PRIMARY KEY,
    user_id        INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    codigo         VARCHAR(50) NOT NULL UNIQUE,   -- CUI
    carrera        VARCHAR(255),
    uses_glasses   INTEGER DEFAULT 0,             -- flags 0/1 sencillos
    uses_cap       INTEGER DEFAULT 0,
    uses_mask      INTEGER DEFAULT 0,
    embedding      TEXT                           -- opcional (no usado por ahora)
  );

  CREATE INDEX idx_students_codigo ON students(codigo);

  -- 3) Cursos (pertenecen a un docente que es un user)
  CREATE TABLE courses (
    id          SERIAL PRIMARY KEY,
    docente_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    nombre      VARCHAR(255) NOT NULL,
    codigo      VARCHAR(50) NOT NULL UNIQUE,
    horario     VARCHAR(255)
  );

  -- 4) Matriculas (N:M entre students y courses) con restricción de unicidad
  CREATE TABLE enrollments (
    id             SERIAL PRIMARY KEY,
    estudiante_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    curso_id       INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    CONSTRAINT uq_enrollment_student_course UNIQUE (estudiante_id, curso_id)
  );

  -- 5) Asistencia
  -- Estados: presente | ausente (CHECK para emular Enum)
  -- fecha_hora: la app guarda hora local (Perú) como naive (sin tz)
  CREATE TABLE attendance (
    id             SERIAL PRIMARY KEY,
    estudiante_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    curso_id       INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    fecha_hora     TIMESTAMP NOT NULL DEFAULT NOW(),
    estado         VARCHAR(20) NOT NULL CHECK (estado IN ('presente','ausente'))
  );

  CREATE INDEX idx_attendance_fecha ON attendance(fecha_hora);
  CREATE INDEX idx_attendance_curso ON attendance(curso_id);
  CREATE INDEX idx_attendance_estudiante ON attendance(estudiante_id);
  ```


Luego, ajusta `backend/.env` para que `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` coincidan. Al iniciar el backend por primera vez, SQLAlchemy creará las tablas.

## Uso rápido

1) Registra usuarios (docente/estudiante), crea curso(s) y matricula estudiantes (o usa auto-matrícula por código de curso).
2) Como estudiante, captura imágenes por categoría:
   - StudentDashboard → "Capturar fotos" → elige categoría y cantidad
   - Se guardan en `dataset/{CUI}/{categoria}/CUI_###.jpg`
3) Como profesor, entrena el modelo:
   - ProfessorDashboard → "Entrenar modelo" (usa cache `*.jpg.npy` para acelerar)
4) Reconocimiento:
   - Streaming (opcional): "Conectar WS" (solo predicciones, no marca)
   - Demo (marca): "Iniciar reconocimiento (demo)" (reemplaza asistencia del día del curso con presentes/ausentes y hora local)

## Detalles técnicos

- Detección: MTCNN (P-Net/R-Net/O-Net) sobre frames de webcam
- Embeddings: `keras-facenet` (vector ~512-D por rostro) con cache por imagen en `*.jpg.npy`
- Clasificación: SVM lineal multi-clase (`SVC(kernel='linear', probability=True)`) + `LabelEncoder`
- Umbral de aceptación configurable (`CONFIDENCE_THRESHOLD`): más alto = más estricto
- Hora de asistencia: local de Perú (o fallback hora local si falta tzdata)

## Estructura de datos

- `dataset/{CUI}/{categoria}/CUI_###.jpg` y `CUI_###.jpg.npy` (cache de embedding)
- `models/face_svm.pkl` y `models/label_encoder.pkl`
- Tablas DB: `users`, `students`, `courses`, `enrollments`, `attendance`

## Solución de problemas

- 401 Unauthorized: reloguear (token expirado) y confirma que el cliente envía `Authorization: Bearer <token>`
- CORS: backend permite `http://localhost:5173` y `http://127.0.0.1:5173`
- tzdata: si ves `ZoneInfoNotFoundError`, `pip install tzdata` y reinicia backend
- Cámara: ajusta `CAMERA_INDEX` en `.env`

## Roadmap (opcional)

- Modelos por curso (SVM por curso) para entrenamientos más rápidos
- k-NN por embeddings para flujo incremental sin reentrenar clasificador
- Consolidar embeddings por curso en `npz` para reducir IO
- Dockerización y CI/CD

## Licencia

Este proyecto es para fines académicos/demostrativos. Ajusta la licencia según tus necesidades.
