from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from ...deps import get_db, require_role
from ...core.config import settings
from ... import models

import cv2 as cv
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet as KFaceNet
from pickle import load as pkl_load
import pickle
from datetime import datetime, date, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os
import logging


router = APIRouter()

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
logger = logging.getLogger(__name__)

def lima_now() -> datetime:
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo("America/Lima"))
    except Exception:
        pass
    # Fallback: naive local time
    return datetime.now()

@router.post("/capture/{cui}")
def capture_faces_for_student(
    cui: str,
    photos: int = 30,
    categoria: Optional[str] = "normal",  # e.g., "gorro", "lentes", "mascarilla"
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin, models.UserRole.estudiante)),
):
    # Authorization and student lookup
    if user.rol == models.UserRole.estudiante:
        student = db.query(models.Student).filter(models.Student.user_id == user.id).first()
        if not student or student.codigo != cui:
            raise HTTPException(status_code=403, detail="No autorizado para capturar para este CUI")
    else:
        student = db.query(models.Student).filter(models.Student.codigo == cui).first()
        if not student:
            raise HTTPException(status_code=404, detail="Estudiante no encontrado")

    if photos <= 0:
        raise HTTPException(status_code=400, detail="El número de fotos debe ser mayor que 0")

    # dataset/{CUI}/{categoria}/
    safe_cat = (categoria or "normal").strip().replace("/", "_")
    target_dir = DATASET_DIR / cui / safe_cat
    target_dir.mkdir(parents=True, exist_ok=True)

    # Initialize camera and detector
    cap = cv.VideoCapture(int(settings.CAMERA_INDEX))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir la cámara")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    detector = MTCNN()

    # Determine next index to avoid overwriting
    existing = [p.name for p in target_dir.glob(f"{cui}_*.jpg")]
    max_idx = 0
    for name in existing:
        try:
            # expect pattern CUI_###.jpg
            base = name.rsplit('.', 1)[0]
            idx = int(base.split('_')[-1])
            if idx > max_idx:
                max_idx = idx
        except Exception:
            pass

    saved = 0
    attempts = 0
    max_attempts = photos * 20  # generous limit to avoid blocking forever

    try:
        while saved < photos and attempts < max_attempts:
            attempts += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            try:
                detections = detector.detect_faces(rgb)
            except Exception as e:
                logger.exception("MTCNN error during detect_faces: %s", e)
                detections = []
            if not detections:
                continue

            # choose highest confidence face
            det = max(detections, key=lambda d: d.get('confidence', 0))
            x, y, w, h = det.get('box', [0, 0, 0, 0])
            x, y = abs(x), abs(y)
            if w <= 0 or h <= 0:
                continue

            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue

            face_resized = cv.resize(face, (160, 160))
            # save as BGR jpeg
            bgr = cv.cvtColor(face_resized, cv.COLOR_RGB2BGR)
            out_path = target_dir / f"{cui}_{max_idx + saved + 1:03d}.jpg"
            cv.imwrite(str(out_path), bgr)
            saved += 1
    finally:
        cap.release()

    return {
        "message": "Captura finalizada",
        "cui": cui,
        "saved": saved,
        "requested": photos,
        "directory": str(target_dir),
    }


def _load_classifier():
    clf_path = MODELS_DIR / "face_svm.pkl"
    le_path = MODELS_DIR / "label_encoder.pkl"
    if not clf_path.exists() or not le_path.exists():
        raise HTTPException(status_code=400, detail="Modelo no entrenado. Ejecuta /api/train primero.")
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    return clf, le


def _mark_attendance(db: Session, codigo: str, course_id: int):
    stu = db.query(models.Student).filter(models.Student.codigo == codigo).first()
    if not stu:
        return False
    # Avoid duplicates same day/course (simple policy)
    # Use Peru timezone for day boundaries (store as naive local time). Fallback to local if tzdata missing.
    today_start = lima_now().replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
    exists = (
        db.query(models.Attendance)
        .filter(
            models.Attendance.estudiante_id == stu.id,
            models.Attendance.curso_id == course_id,
            models.Attendance.fecha_hora >= today_start,
        )
        .first()
    )
    if exists:
        return False
    rec = models.Attendance(
        estudiante_id=stu.id,
        curso_id=course_id,
        fecha_hora=lima_now().replace(tzinfo=None),
        estado=models.AttendanceState.presente,
    )
    db.add(rec)
    db.commit()
    return True


def _finalize_attendance_for_session(db: Session, course_id: int, recognized_cuis: set[str]):
    """
    Replace today's attendance for the course with the latest session results:
    - PRESENTE for recognized_cuis
    - AUSENTE for enrolled students not recognized
    """
    now_local = lima_now().replace(tzinfo=None)
    today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

    # Remove today's existing attendance for this course
    (
        db.query(models.Attendance)
        .filter(
            models.Attendance.curso_id == course_id,
            models.Attendance.fecha_hora >= today_start,
        )
        .delete(synchronize_session=False)
    )

    # Get enrolled students and their CUI codes
    enrolled = (
        db.query(models.Student)
        .join(models.Enrollment, models.Enrollment.estudiante_id == models.Student.id)
        .filter(models.Enrollment.curso_id == course_id)
        .all()
    )

    present = 0
    absent = 0
    for stu in enrolled:
        estado = models.AttendanceState.presente if stu.codigo in recognized_cuis else models.AttendanceState.ausente
        if estado == models.AttendanceState.presente:
            present += 1
        else:
            absent += 1
        db.add(
            models.Attendance(
                estudiante_id=stu.id,
                curso_id=course_id,
                fecha_hora=now_local,
                estado=estado,
            )
        )
    db.commit()
    return {"present": present, "absent": absent}


@router.get("/recognize/start")
def recognize_start(
    course_id: int,
    duration_minutes: int = 1,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin)),
):
    # Validate course ownership for docente
    course = db.get(models.Course, course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Curso no encontrado")
    if user.rol == models.UserRole.docente and course.docente_id != user.id:
        raise HTTPException(status_code=403, detail="No autorizado para este curso")

    # Load model
    clf, le = _load_classifier()
    embedder = KFaceNet()
    detector = MTCNN()

    # Camera
    cap = cv.VideoCapture(int(settings.CAMERA_INDEX))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir la cámara")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    recognized: list[dict] = []
    recognized_cuis: set[str] = set()
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    try:
        while datetime.now() < end_time:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            try:
                detections = detector.detect_faces(rgb)
            except Exception:
                detections = []
            if not detections:
                continue

            det = max(detections, key=lambda d: d.get('confidence', 0))
            x, y, w, h = det.get('box', [0, 0, 0, 0])
            x, y = abs(x), abs(y)
            if w <= 0 or h <= 0:
                continue

            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face = cv.resize(face, (160, 160))

            # embedding
            # KFaceNet expects RGB images normalized internally
            embed = embedder.embeddings([face])[0]

            # predict
            try:
                probs = clf.predict_proba([embed])[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                label = le.classes_[idx]
            except Exception as e:
                logger.exception("Classifier predict_proba failed, fallback to predict: %s", e)
                # fallback to decision_function if no predict_proba
                pred = clf.predict([embed])[0]
                label = pred
                conf = 1.0

            if conf >= settings.CONFIDENCE_THRESHOLD:
                recognized_cuis.add(str(label))
                recognized.append({"codigo": label, "confidence": conf})
    finally:
        cap.release()

    # Replace today's attendance with this session results
    summary = _finalize_attendance_for_session(db, course_id, recognized_cuis)
    return {
        "message": "Reconocimiento finalizado",
        "recognized": recognized,
        "course_id": course_id,
        "session_summary": summary,
    }

@router.post("/train")
def train_model_endpoint(
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin)),
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_DIR.exists():
        raise HTTPException(status_code=400, detail="No hay dataset para entrenar")

    embedder = KFaceNet()
    X: list[np.ndarray] = []
    y: list[str] = []

    IMAGE_EXTS = (".jpg", ".jpeg", ".png")
    # Recorre dataset/{CUI}/**/*.jpg y cachea embeddings en archivos .npy al lado de cada imagen
    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        for root, _dirs, files in os.walk(person_dir):
            for fname in files:
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                img_path = Path(root) / fname
                emb_path = img_path.with_suffix(img_path.suffix + ".npy")  # ej.: foto.jpg.npy
                emb: Optional[np.ndarray] = None
                try:
                    if emb_path.exists() and emb_path.stat().st_mtime >= img_path.stat().st_mtime:
                        emb = np.load(str(emb_path))
                except Exception as e:
                    logger.exception("No se pudo cargar embedding cacheado %s: %s", emb_path, e)
                    emb = None
                if emb is None:
                    img = cv.imread(str(img_path))
                    if img is None:
                        continue
                    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (160, 160):
                        rgb = cv.resize(rgb, (160, 160))
                    emb = embedder.embeddings([rgb])[0]
                    try:
                        np.save(str(emb_path), emb)
                    except Exception as e:
                        logger.exception("No se pudo guardar embedding en %s: %s", emb_path, e)
                X.append(emb)
                y.append(label)

    if not X:
        raise HTTPException(status_code=400, detail="Dataset vacío. Capture rostros antes de entrenar")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y_enc)

    with open(MODELS_DIR / "face_svm.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return {"message": "Modelo entrenado", "classes": list(le.classes_), "samples": len(X)}


@router.get("/model/status")
def model_status():
    clf_path = MODELS_DIR / "face_svm.pkl"
    le_path = MODELS_DIR / "label_encoder.pkl"
    trained = clf_path.exists() and le_path.exists()
    classes: list[str] = []
    mtime: Optional[float] = None
    if trained:
        try:
            with open(le_path, "rb") as f:
                le = pickle.load(f)
                classes = list(getattr(le, 'classes_', []))
            mtime = max(clf_path.stat().st_mtime, le_path.stat().st_mtime)
        except Exception as e:
            logger.exception("Error leyendo modelo: %s", e)
    warnings = []
    if settings.JWT_SECRET == "CHANGE_ME_SUPER_SECRET":
        warnings.append("JWT_SECRET utiliza el valor por defecto. Cambiar en backend/.env")
    return {
        "trained": trained,
        "classes": classes,
        "modified_at": mtime,
        "threshold": settings.CONFIDENCE_THRESHOLD,
        "warnings": warnings,
    }


@router.websocket("/recognize/ws")
async def recognize_ws(websocket: WebSocket, course_id: int):
    await websocket.accept()
    try:
        clf, le = _load_classifier()
    except HTTPException as e:
        await websocket.send_json({"error": e.detail})
        await websocket.close()
        return

    embedder = KFaceNet()
    detector = MTCNN()
    cap = cv.VideoCapture(int(settings.CAMERA_INDEX))
    if not cap.isOpened():
        await websocket.send_json({"error": "No se pudo abrir la cámara"})
        await websocket.close()
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                await websocket.send_json({"event": "frame_error"})
                continue
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            try:
                detections = detector.detect_faces(rgb)
            except Exception:
                detections = []
            if not detections:
                await websocket.send_json({"event": "no_face"})
                continue
            det = max(detections, key=lambda d: d.get('confidence', 0))
            x, y, w, h = det.get('box', [0, 0, 0, 0])
            x, y = abs(x), abs(y)
            if w <= 0 or h <= 0:
                await websocket.send_json({"event": "invalid_box"})
                continue
            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                await websocket.send_json({"event": "empty_crop"})
                continue
            face = cv.resize(face, (160, 160))
            emb = embedder.embeddings([face])[0]
            try:
                probs = clf.predict_proba([emb])[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                label = le.classes_[idx]
            except Exception:
                pred = clf.predict([emb])[0]
                label = pred
                conf = 1.0
            await websocket.send_json({"event": "prediction", "codigo": label, "confidence": conf})
    finally:
        cap.release()
        await websocket.close()
