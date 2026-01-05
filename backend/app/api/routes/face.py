from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound
from ...deps import get_db, require_role
from ...core.config import settings
from ... import models

import cv2 as cv
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet as KFaceNet
from pickle import load as pkl_load, dump as pkl_dump
import pickle
from datetime import datetime, date, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    import pytz
    ZoneInfo = pytz.timezone  # type: ignore
import os
import logging
import faiss
from sklearn.preprocessing import LabelEncoder
import json
import base64
from PIL import Image, ImageDraw, ImageFont


router = APIRouter()

# Configurar el logger primero para asegurar que los mensajes se muestren
logger = logging.getLogger(__name__)

# Función para encontrar el directorio raíz del proyecto
def find_project_root():
    """
    Busca el directorio raíz del proyecto (donde está el directorio 'backend')
    """
    current = Path(__file__).resolve()
    # Subir hasta encontrar el directorio que contiene 'backend'
    while current.parent != current:
        if (current / 'backend').is_dir() and (current / 'frontend').is_dir():
            return current
        current = current.parent
    return Path.cwd()  # Fallback al directorio actual

# Obtener rutas absolutas
PROJECT_ROOT = find_project_root()
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"

# Asegurarse de que los directorios existan
DATASET_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Log para depuración
try:
    logger.info(f"Directorio raíz del proyecto: {PROJECT_ROOT}")
    logger.info(f"Ruta del dataset: {DATASET_DIR}")
    logger.info(f"Ruta de modelos: {MODELS_DIR}")
    logger.info(f"¿Existe el dataset?: {DATASET_DIR.exists()}")
    logger.info(f"¿Existe el directorio de modelos?: {MODELS_DIR.exists()}")
    
    # Listar contenido del dataset para depuración
    if DATASET_DIR.exists():
        logger.info(f"Contenido del dataset: {[p.name for p in DATASET_DIR.iterdir() if p.is_dir()]}")
        for subdir in DATASET_DIR.iterdir():
            if subdir.is_dir():
                logger.info(f"  {subdir.name}: {len(list(subdir.glob('*.*')))} archivos")
    else:
        logger.warning(f"El directorio del dataset no existe: {DATASET_DIR}")
        
except Exception as e:
    logger.error(f"Error al configurar rutas: {str(e)}", exc_info=True)
    raise


MIN_FACE_CONF = 0.70  # Umbral mínimo para marcar asistencia (equivale a 80% voto + distancia)
MAX_L2_DISTANCE = 1.20  # Distancia máxima aceptable (menor = más similar)

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


def _load_classifier(course_id: Optional[int] = None):
    """
    Carga el modelo FAISS y sus componentes necesarios.
    
    Args:
        course_id: ID del curso. Si se proporciona, carga el modelo específico del curso.
                   Si es None, intenta cargar el modelo global (compatibilidad hacia atrás).
    
    Returns:
        dict: Diccionario con los componentes del modelo FAISS:
            - type: "faiss"
            - index: Índice FAISS cargado
            - encoder: Codificador de etiquetas
            - embeddings: Array con los embeddings de entrenamiento
            - labels: Array con las etiquetas correspondientes a los embeddings
            - course_id: ID del curso (si se proporcionó)
            
    Raises:
        HTTPException: Si no se encuentra el modelo o hay un error al cargarlo
    """
    # Determinar la ruta del modelo según si hay course_id
    if course_id is not None:
        course_models_dir = MODELS_DIR / "courses" / f"curso_{course_id}"
        faiss_index_path = course_models_dir / "face-recognition-faiss.index"
        le_path = course_models_dir / "label-encoder-faiss.pkl"
        embeddings_path = course_models_dir / "face-embeddings-faiss.npz"
        model_type = f"curso_{course_id}"
    else:
        # Compatibilidad: modelo global
        faiss_index_path = MODELS_DIR / "face-recognition-faiss.index"
        le_path = MODELS_DIR / "label-encoder-faiss.pkl"
        embeddings_path = MODELS_DIR / "face-embeddings-faiss.npz"
        model_type = "global"
    
    # Verificar que existan todos los archivos necesarios
    missing_files = []
    if not faiss_index_path.exists():
        missing_files.append(str(faiss_index_path.name))
    if not le_path.exists():
        missing_files.append(str(le_path.name))
    if not embeddings_path.exists():
        missing_files.append(str(embeddings_path.name))
        
    if missing_files:
        if course_id is not None:
            raise HTTPException(
                status_code=400, 
                detail=f"Modelo no encontrado para el curso {course_id}. Archivos faltantes: {', '.join(missing_files)}. "
                       f"Ejecuta /api/train?course_id={course_id} primero."
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Archivos del modelo FAISS no encontrados: {', '.join(missing_files)}. Ejecuta /api/train primero."
            )
    
    try:
        # Cargar el índice FAISS
        logger.info(f"Cargando índice FAISS ({model_type})...")
        index = faiss.read_index(str(faiss_index_path))
        
        # Cargar el codificador de etiquetas
        logger.info(f"Cargando codificador de etiquetas ({model_type})...")
        with open(le_path, "rb") as f:
            le = pickle.load(f)
            
        # Cargar los embeddings y etiquetas para referencia
        logger.info(f"Cargando embeddings y etiquetas ({model_type})...")
        data = np.load(embeddings_path, allow_pickle=True)
        
        # Verificar que los datos estén en el formato correcto
        if 'X' not in data or 'y' not in data:
            raise ValueError("Los datos de embeddings no tienen el formato esperado (X, y)")
            
        logger.info(f"Modelo FAISS ({model_type}) cargado correctamente con {len(data['y'])} muestras")
        
        result = {
            "type": "faiss",
            "index": index,
            "encoder": le,
            "embeddings": data['X'],
            "labels": data['y']
        }
        
        if course_id is not None:
            result["course_id"] = course_id
            
        return result
        
    except Exception as e:
        error_msg = f"Error al cargar el modelo FAISS ({model_type}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=error_msg
        )


def _mark_attendance(db: Session, codigo: str, course_id: int):
    """Marca la asistencia de un estudiante si supera el umbral de confianza"""
    # Obtener el estudiante
    student = db.query(models.Student).filter(models.Student.codigo == codigo).first()
    if not student:
        logger.warning(f"Estudiante con código {codigo} no encontrado")
        return
    # Verificar si ya existe un registro de asistencia hoy
    today = datetime.utcnow().date()
    existing = db.query(models.Attendance).filter(
        models.Attendance.estudiante_id == student.id,
        models.Attendance.curso_id == course_id,
        models.Attendance.fecha_hora >= today,
        models.Attendance.fecha_hora < today + timedelta(days=1)
    ).first()
    if not existing:
        # Crear nuevo registro de asistencia
        attendance = models.Attendance(
            estudiante_id=student.id,
            curso_id=course_id,
            fecha_hora=datetime.utcnow(),
            estado=models.AttendanceState.presente
        )
        db.add(attendance)
        db.commit()
        logger.info(f"Marcando asistencia para {codigo} en curso {course_id}")


def _finalize_attendance_for_session(db: Session, course_id: int, recognized_cuis: set[str]):
    """No marca ausentes automáticamente; solo devuelve el resumen de reconocidos."""
    enrolled_students = db.query(models.Student).join(
        models.Enrollment,
        models.Enrollment.estudiante_id == models.Student.id
    ).filter(
        models.Enrollment.curso_id == course_id
    ).all()

    today = datetime.utcnow().date()
    attendance_records = (
        db.query(models.Attendance)
        .filter(
            models.Attendance.curso_id == course_id,
            models.Attendance.fecha_hora >= today,
            models.Attendance.fecha_hora < today + timedelta(days=1)
        )
        .all()
    )

    present_student_ids = {att.estudiante_id for att in attendance_records if att.estado == models.AttendanceState.presente}
    absent_candidates = []
    for stu in enrolled_students:
        if stu.id not in present_student_ids:
            absent_candidates.append(stu.codigo)

    summary = {
        "recognized": sorted(list(recognized_cuis)),
        "present_count": len(present_student_ids),
        "absent_candidates": absent_candidates
    }
    return summary


@router.get("/recognize/start")
def recognize_start(
    course_id: int,
    duration_minutes: int = 1,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin)),
):
    """
    Inicia el reconocimiento facial para el curso especificado.
    Similar a main.py pero integrado con FastAPI.
    """
    # Validar curso
    course = db.get(models.Course, course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Curso no encontrado")
    if user.rol == models.UserRole.docente and course.docente_id != user.id:
        raise HTTPException(status_code=403, detail="No autorizado para este curso")

    # Obtener estudiantes matriculados en el curso
    enrolled_students = (
        db.query(models.Student)
        .join(models.Enrollment, models.Enrollment.estudiante_id == models.Student.id)
        .filter(models.Enrollment.curso_id == course_id)
        .all()
    )
    
    if not enrolled_students:
        raise HTTPException(status_code=400, detail="No hay estudiantes matriculados en este curso")
    
    # Obtener códigos (CUI) de estudiantes matriculados
    enrolled_cuis = {stu.codigo for stu in enrolled_students}
    logger.info(f"Estudiantes matriculados en el curso {course_id}: {sorted(enrolled_cuis)}")

    # Limpiar asistencias del día para este curso antes de iniciar nueva sesión
    today = datetime.utcnow().date()
    db.query(models.Attendance).filter(
        models.Attendance.curso_id == course_id,
        models.Attendance.fecha_hora >= today,
        models.Attendance.fecha_hora < today + timedelta(days=1)
    ).delete(synchronize_session=False)
    db.commit()

    # Cargar el modelo FAISS específico del curso
    model_info = _load_classifier(course_id=course_id)
    if model_info["type"] != "faiss":
        raise HTTPException(status_code=500, detail="Tipo de modelo no soportado")
    
    # Inicializar componentes
    embedder = KFaceNet()
    detector = MTCNN()

    # Configurar la cámara
    cap = cv.VideoCapture(int(settings.CAMERA_INDEX))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir la cámara")
    
    # Configuración de la cámara
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    recognized: list[dict] = []
    recognized_cuis: set[str] = set()
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    try:
        frame_counter = 0
        while datetime.now() < end_time:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
                
            # Hacer una copia del frame para mostrar
            display_frame = frame.copy()
            
            # Convertir a RGB para MTCNN
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Procesar solo 1 de cada 3 frames para mejorar el rendimiento
            frame_counter += 1
            if frame_counter % 3 != 0:
                continue
            
            # Detectar rostros
            try:
                detections = detector.detect_faces(rgb)
            except Exception as e:
                logger.error(f"Error en detección de rostros: {e}")
                detections = []
                
            # Procesar cada detección
            for det in detections:
                # Filtro de calidad de detección para reducir falsos positivos
                det_conf = float(det.get('confidence', 0) or 0)
                x, y, w, h = det.get('box', [0, 0, 0, 0])
                x, y = abs(x), abs(y)
                if w <= 0 or h <= 0:
                    continue
                if det_conf < 0.90:
                    continue
                if (w * h) < (60 * 60):
                    continue

                face = rgb[y:y+h, x:x+w]
                if face.size == 0:
                    continue
                    
                # Redimensionar y extraer embedding
                face_resized = cv.resize(face, (160, 160))
                emb = embedder.embeddings([face_resized])[0].astype('float32')
                
                # Búsqueda de vecinos más cercanos con FAISS
                faiss.normalize_L2(emb.reshape(1, -1))
                k = 5  # Número de vecinos a buscar
                distances, indices = model_info["index"].search(emb.reshape(1, -1), k)
                
                # Obtener etiquetas de los vecinos
                neighbor_labels = model_info["labels"][indices[0]]
                
                # Contar ocurrencias de cada etiqueta
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                
                if len(counts) > 0:  # Si hay etiquetas
                    most_common_idx = np.argmax(counts)
                    label = str(unique_labels[most_common_idx])
                    
                    # Calcular confianza basada en la mayoría de votos
                    conf = counts[most_common_idx] / k
                    
                    # Ajustar confianza basada en la distancia
                    if distances[0][0] > 0:  # Evitar división por cero
                        distance_confidence = 1.0 / (1.0 + distances[0][0])
                        conf = 0.7 * conf + 0.3 * distance_confidence
                    
                    # Verificar si el estudiante reconocido pertenece al curso
                    is_enrolled = label in enrolled_cuis
                    
                    # Registrar reconocimiento si supera el umbral Y pertenece al curso
                    if (conf >= MIN_FACE_CONF and is_enrolled) and distances[0][0] <= MAX_L2_DISTANCE:
                        recognized_cuis.add(label)
                        recognized.append({"codigo": label, "confidence": float(conf)})
                        
                        # Marcar asistencia
                        _mark_attendance(db, label, course_id)
                    elif conf >= MIN_FACE_CONF and not is_enrolled and distances[0][0] <= MAX_L2_DISTANCE:
                        # Reconocido pero no pertenece al curso - marcar como desconocido
                        logger.warning(f"Estudiante {label} reconocido pero no está matriculado en el curso {course_id}")
                        recognized.append({"codigo": "Desconocido", "confidence": float(conf), "original_label": label})
                
            # Mostrar el frame con las detecciones (solo si no estamos en modo headless)
            if not os.environ.get('OPENCV_HEADLESS'):
                for det in detections:
                    x, y, w, h = det.get('box', [0, 0, 0, 0])
                    x, y = abs(x), abs(y)
                    if w > 0 and h > 0:
                        # Buscar la etiqueta y confianza para esta detección
                        face = rgb[y:y+h, x:x+w]
                        if face.size == 0:
                            continue
                            
                        face_resized = cv.resize(face, (160, 160))
                        emb = embedder.embeddings([face_resized])[0].astype('float32')
                        
                        # Búsqueda de vecinos más cercanos con FAISS
                        faiss.normalize_L2(emb.reshape(1, -1))
                        k = 5
                        distances, indices = model_info["index"].search(emb.reshape(1, -1), k)
                        neighbor_labels = model_info["labels"][indices[0]]
                        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                        
                        if len(counts) > 0:
                            most_common_idx = np.argmax(counts)
                            label = str(unique_labels[most_common_idx])
                            conf = counts[most_common_idx] / k
                            
                            # Ajustar confianza basada en la distancia
                            if distances[0][0] > 0:
                                distance_confidence = 1.0 / (1.0 + distances[0][0])
                                conf = 0.7 * conf + 0.3 * distance_confidence
                            
                            # Verificar si el estudiante reconocido pertenece al curso
                            is_enrolled = label in enrolled_cuis
                            
                            # Determinar color y etiqueta basado en confianza y pertenencia al curso
                            if conf >= settings.CONFIDENCE_THRESHOLD and is_enrolled:
                                color = (0, 255, 0)  # Verde para reconocido y matriculado
                                display_label = label
                            elif conf >= settings.CONFIDENCE_THRESHOLD and not is_enrolled:
                                color = (255, 165, 0)  # Naranja para reconocido pero no matriculado
                                display_label = "Desconocido"
                            else:
                                color = (0, 0, 255)  # Rojo para no reconocido
                                display_label = "Desconocido"
                            
                            # Dibujar rectángulo
                            cv.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                            
                            # Mostrar etiqueta y confianza
                            label_text = f"{display_label} ({conf*100:.1f}%)"
                            cv.putText(display_frame, label_text, (x, y-10), 
                                     cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                     
                            # Forzar actualización de la ventana
                            cv.imshow('Reconocimiento Facial', display_frame)
                            cv.waitKey(1)
                        else:
                            # Si no se reconoce la cara
                            cv.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv.putText(display_frame, "Desconocido", (x, y-10), 
                                     cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                # Mostrar el frame con todas las detecciones
                cv.imshow('Reconocimiento Facial', display_frame)
                
                # Salir si se presiona 'q'
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except Exception as e:
        logger.error(f"Error durante el reconocimiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error durante el reconocimiento: {str(e)}")
    
    finally:
        cap.release()
        if not os.environ.get('OPENCV_HEADLESS'):
            cv.destroyAllWindows()

    # Finalizar la sesión de reconocimiento
    summary = _finalize_attendance_for_session(db, course_id, recognized_cuis)
    
    # Obtener información detallada de asistencia para cada estudiante
    today_start = lima_now().replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
    attendance_records = (
        db.query(models.Attendance)
        .filter(
            models.Attendance.curso_id == course_id,
            models.Attendance.fecha_hora >= today_start,
        )
        .all()
    )
    
    # Crear lista detallada de asistencia por estudiante
    attendance_details = []
    enrolled_students_dict = {stu.id: stu for stu in enrolled_students}
    
    for att in attendance_records:
        student = enrolled_students_dict.get(att.estudiante_id)
        if student:
            attendance_details.append({
                "estudiante_id": att.estudiante_id,
                "codigo": student.codigo,
                "nombre": f"{student.user.nombres} {student.user.apellidos}" if student.user else student.codigo,
                "estado": att.estado.value,
                "fecha_hora": att.fecha_hora.isoformat() if att.fecha_hora else None,
            })
    
    # Identificar estudiantes ausentes (matriculados pero sin asistencia)
    present_student_ids = {att.estudiante_id for att in attendance_records if att.estado == models.AttendanceState.presente}
    for stu in enrolled_students:
        if stu.id not in present_student_ids:
            attendance_details.append({
                "estudiante_id": stu.id,
                "codigo": stu.codigo,
                "nombre": f"{stu.user.nombres} {stu.user.apellidos}" if stu.user else stu.codigo,
                "estado": "ausente",
                "fecha_hora": None,
            })
    
    return {
        "message": "Reconocimiento finalizado",
        "recognized": recognized,
        "course_id": course_id,
        "course_name": course.nombre,
        "session_summary": summary,
        "attendance_details": attendance_details,
        "recognized_count": len(recognized_cuis),
        "total_students": len(enrolled_students),
    }

@router.post("/train")
def train_model_endpoint(
    course_id: int,
    course_name: Optional[str] = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_role(models.UserRole.docente, models.UserRole.admin)),
):
    """
    Entrena un modelo de reconocimiento facial específico para un curso.
    Solo incluye estudiantes matriculados en ese curso.
    """
    # Validar curso
    course = db.get(models.Course, course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Curso no encontrado")
    
    # Verificar permisos
    if user.rol == models.UserRole.docente and course.docente_id != user.id:
        raise HTTPException(status_code=403, detail="No autorizado para este curso")
    
    # Obtener estudiantes matriculados en el curso
    enrolled_students = (
        db.query(models.Student)
        .join(models.Enrollment, models.Enrollment.estudiante_id == models.Student.id)
        .filter(models.Enrollment.curso_id == course_id)
        .all()
    )
    
    if not enrolled_students:
        raise HTTPException(status_code=400, detail="No hay estudiantes matriculados en este curso")
    
    # Obtener códigos (CUI) de estudiantes matriculados
    enrolled_cuis = {stu.codigo for stu in enrolled_students}
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_DIR.exists():
        raise HTTPException(status_code=400, detail="No hay dataset para entrenar")

    # Crear directorio específico para el curso
    course_models_dir = MODELS_DIR / "courses" / f"curso_{course_id}"
    course_models_dir.mkdir(parents=True, exist_ok=True)

    embedder = KFaceNet()
    X: list[np.ndarray] = []
    y: list[str] = []

    IMAGE_EXTS = (".jpg", ".jpeg", ".png")
    # Recorre dataset/{CUI}/**/*.jpg solo para estudiantes del curso
    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        label = person_dir.name  # El nombre del directorio es el CUI
        
        # Filtrar solo estudiantes matriculados en el curso
        if label not in enrolled_cuis:
            logger.info(f"Omitiendo {label} - no está matriculado en el curso {course_id}")
            continue
        
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
        raise HTTPException(
            status_code=400, 
            detail=f"Dataset vacío para el curso. Capture rostros de los estudiantes matriculados antes de entrenar. "
                   f"Estudiantes esperados: {', '.join(sorted(enrolled_cuis))}"
        )

    # Convertir a numpy array
    X_np = np.array(X).astype('float32')
    
    # Normalizar los embeddings para mejorar el rendimiento de L2
    faiss.normalize_L2(X_np)
    
    # Crear y entrenar el índice FAISS
    d = X_np.shape[1]  # Dimensión de los embeddings
    index = faiss.IndexFlatL2(d)  # Usar distancia L2 (Euclidiana)
    index.add(X_np)  # Añadir todos los vectores al índice
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Guardar el modelo FAISS en el directorio del curso
    faiss.write_index(index, str(course_models_dir / "face-recognition-faiss.index"))
    with open(course_models_dir / "label-encoder-faiss.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # Guardar los embeddings y etiquetas para referencia
    np.savez_compressed(
        str(course_models_dir / "face-embeddings-faiss.npz"), 
        X=X_np, 
        y=np.array(y)
    )
    
    # Calcular precisión aproximada en entrenamiento (opcional)
    _, indices = index.search(X_np, 5)  # Buscar 5 vecinos más cercanos
    correct = 0
    for i, neighbors in enumerate(indices):
        neighbor_labels = y_enc[neighbors]
        if y_enc[i] in neighbor_labels:
            correct += 1
    accuracy = correct / len(y_enc)
    
    unique_students = len(set(y))
    
    return {
        "message": f"Modelo FAISS entrenado exitosamente para el curso '{course.nombre}'",
        "course_id": course_id,
        "course_name": course.nombre,
        "samples": len(X_np),
        "students_count": unique_students,
        "dimensions": d,
        "training_accuracy": accuracy,
        "index_size": index.ntotal,
        "model_path": str(course_models_dir)
    }


@router.get("/model/status")
def model_status():
    # Verificar si existe el modelo FAISS
    faiss_index_path = MODELS_DIR / "face-recognition-faiss.index"
    le_path = MODELS_DIR / "label-encoder-faiss.pkl"
    embeddings_path = MODELS_DIR / "face-embeddings-faiss.npz"
    
    faiss_trained = all([faiss_index_path.exists(), le_path.exists(), embeddings_path.exists()])
    
    # Verificar si existe el modelo SVM antiguo (para compatibilidad)
    svm_path = MODELS_DIR / "face_svm.pkl"
    old_le_path = MODELS_DIR / "label_encoder.pkl"
    svm_trained = svm_path.exists() and old_le_path.exists()
    
    model_info = {
        "model_type": "faiss" if faiss_trained else "svm" if svm_trained else None,
        "trained": faiss_trained or svm_trained,
        "classes": [],
        "last_trained": None,
        "threshold": settings.CONFIDENCE_THRESHOLD,
        "samples": 0,
        "dimensions": 0,
        "warnings": []
    }


@router.websocket("/recognize/ws")
async def recognize_ws(websocket: WebSocket, course_id: int):
    await websocket.accept()
    
    # Validar curso y obtener estudiantes matriculados
    from ...db.session import SessionLocal
    db_local = SessionLocal()
    try:
        course = db_local.get(models.Course, course_id)
        if not course:
            await websocket.send_json({"error": "Curso no encontrado"})
            await websocket.close()
            return
        
        # Obtener estudiantes matriculados en el curso
        enrolled_students = (
            db_local.query(models.Student)
            .join(models.Enrollment, models.Enrollment.estudiante_id == models.Student.id)
            .filter(models.Enrollment.curso_id == course_id)
            .all()
        )
        
        enrolled_cuis = {stu.codigo for stu in enrolled_students}
        logger.info(f"WebSocket: Estudiantes matriculados en el curso {course_id}: {sorted(enrolled_cuis)}")
    except Exception as e:
        logger.error(f"Error al validar curso en WebSocket: {e}")
        await websocket.send_json({"error": f"Error al validar curso: {str(e)}"})
        await websocket.close()
        return
    finally:
        db_local.close()
    
    # Cargar el modelo FAISS específico del curso
    try:
        model_info = _load_classifier(course_id=course_id)
        if model_info["type"] != "faiss":
            raise HTTPException(status_code=500, detail="Tipo de modelo no soportado")
    except HTTPException as e:
        await websocket.send_json({"error": e.detail})
        await websocket.close()
        return
    
    # Inicializar componentes comunes
    embedder = KFaceNet()
    detector = MTCNN()
    
    # Configurar la cámara
    cap = cv.VideoCapture(int(settings.CAMERA_INDEX))
    if not cap.isOpened():
        await websocket.send_json({"error": "No se pudo abrir la cámara"})
        await websocket.close()
        return
    
    # Configurar resolución de la cámara
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Contador de frames para no procesar todos los frames
    frame_counter = 0
    
    try:
        while True:
            try:
                # Recibir mensaje del cliente (podría usarse para control)
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
                
            # Leer frame de la cámara
            ok, frame = cap.read()
            if not ok or frame is None:
                await websocket.send_json({"event": "frame_error"})
                continue
                
            # Convertir a RGB para MTCNN y FaceNet
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Detectar rostros (usar solo cada 3 frames para mejorar rendimiento)
            frame_counter += 1
            if frame_counter % 3 != 0:
                continue
                
            try:
                detections = detector.detect_faces(rgb)
            except Exception as e:
                logger.error(f"Error en detección de rostros: {e}")
                detections = []
                
            if not detections:
                await websocket.send_json({"event": "no_face"})
                continue
                
            # Tomar el rostro con mayor confianza
            det = max(detections, key=lambda d: d.get('confidence', 0))
            det_conf = float(det.get('confidence', 0) or 0)
            x, y, w, h = det.get('box', [0, 0, 0, 0])
            x, y = abs(x), abs(y)
            
            if w <= 0 or h <= 0:
                await websocket.send_json({"event": "invalid_box"})
                continue
            if det_conf < 0.90:
                await websocket.send_json({"event": "low_conf_face"})
                continue
            if (w * h) < (60 * 60):
                await websocket.send_json({"event": "small_face"})
                continue
                
            # Extraer el rostro
            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                await websocket.send_json({"event": "empty_crop"})
                continue
                
            # Redimensionar y extraer embedding
            face = cv.resize(face, (160, 160))
            emb = embedder.embeddings([face])[0].astype('float32')
            
            try:
                # Usar FAISS para la búsqueda de vecinos más cercanos
                faiss.normalize_L2(emb.reshape(1, -1))  # Normalizar el embedding
                
                # Buscar los 5 vecinos más cercanos
                k = 5
                distances, indices = model_info["index"].search(emb.reshape(1, -1), k)
                
                # Obtener las etiquetas de los vecinos
                neighbor_labels = model_info["labels"][indices[0]]
                
                # Contar ocurrencias de cada etiqueta
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                
                # Obtener la etiqueta más común
                most_common_idx = np.argmax(counts)
                label = str(unique_labels[most_common_idx])
                
                # Calcular confianza basada en la mayoría de votos
                conf = counts[most_common_idx] / k
                
                # Ajustar la confianza basada en la distancia
                # Cuanto menor sea la distancia, mayor será la confianza
                if distances[0][0] > 0:  # Evitar división por cero
                    distance_confidence = 1.0 / (1.0 + distances[0][0])
                    conf = 0.7 * conf + 0.3 * distance_confidence
                
                # Verificar si el estudiante reconocido pertenece al curso
                is_enrolled = label in enrolled_cuis
                
                # Determinar el código a enviar
                attendance_marked = False
                if (conf >= MIN_FACE_CONF and is_enrolled) and distances[0][0] <= MAX_L2_DISTANCE:
                    # Reconocido y matriculado - enviar código del estudiante
                    final_codigo = label
                    try:
                        _mark_attendance(db_live, label, course_id)
                        attendance_marked = True
                    except Exception as e:
                        logger.error(f"WebSocket: No se pudo marcar asistencia para {label}: {e}")
                elif conf >= MIN_FACE_CONF and not is_enrolled and distances[0][0] <= MAX_L2_DISTANCE:
                    # Reconocido pero no matriculado - marcar como desconocido
                    final_codigo = "Desconocido"
                    logger.warning(f"WebSocket: Estudiante {label} reconocido pero no está matriculado en el curso {course_id}")
                else:
                    # No reconocido con suficiente confianza
                    final_codigo = "Desconocido"
                
                # Enviar la predicción al cliente
                await websocket.send_json({
                    "event": "prediction", 
                    "codigo": final_codigo, 
                    "confidence": float(conf),
                    "model_type": "faiss",
                    "is_enrolled": is_enrolled if conf >= settings.CONFIDENCE_THRESHOLD else False
                })
                if attendance_marked:
                    await websocket.send_json({
                        "event": "attendance_marked",
                        "codigo": final_codigo,
                        "confidence": float(conf)
                    })
                
            except Exception as e:
                logger.error(f"Error en la predicción: {e}")
                await websocket.send_json({
                    "event": "prediction_error",
                    "error": str(e)
                })
    finally:
        cap.release()
        db_live.close()
        await websocket.close()
