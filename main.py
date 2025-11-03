import cv2 as cv
import numpy as np
import os
import faiss
import pickle
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet

# Configuración de entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Inicializar FaceNet
facenet = FaceNet()

# Cargar el modelo FAISS
def load_faiss_model():
    print("Cargando modelo FAISS...")
    # Cargar el índice FAISS
    index = faiss.read_index("face-recognition-faiss.index")
    
    # Cargar el codificador de etiquetas
    with open('label-encoder-faiss.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    # Cargar los embeddings y etiquetas para referencia
    data = np.load('face-embeddings-faiss.npz', allow_pickle=True)
    
    print("✅ Modelo FAISS cargado correctamente")
    return index, encoder, data

# Cargar el modelo
index, encoder, data = load_faiss_model()

# Inicializar el detector de caras de OpenCV
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Configurar la cámara
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# Parámetros de k-NN
K = 5  # Número de vecinos a considerar

def predict_face(face_img):
    """Predice la identidad de un rostro usando FAISS"""
    # Preprocesar la imagen
    face_img = cv.resize(face_img, (160, 160))
    face_img = np.expand_dims(face_img, axis=0)
    
    # Extraer embedding con FaceNet
    embedding = facenet.embeddings(face_img).astype('float32')
    faiss.normalize_L2(embedding)  # Normalizar para consistencia
    
    # Buscar los K vecinos más cercanos
    distances, indices = index.search(embedding, K)
    
    # Obtener las etiquetas de los vecinos
    neighbor_labels = data['labels'][indices[0]]
    
    # Contar ocurrencias de cada etiqueta
    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
    
    # Obtener la etiqueta más común
    most_common_label = unique_labels[np.argmax(counts)]
    confidence = np.max(counts) / K  # Porcentaje de votos para la etiqueta
    
    return most_common_label, confidence

# Bucle principal
print("Iniciando reconocimiento facial. Presiona 'q' para salir.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame")
        break
    
    # Convertir a RGB para FaceNet y a escala de grises para el detector de caras
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detectar caras
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extraer el rostro
        face_img = rgb_img[y:y+h, x:x+w]
        
        try:
            # Predecir la identidad
            identity, confidence = predict_face(face_img)
            
            # Dibujar el rectángulo y la etiqueta
            color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)  # Verde si alta confianza, naranja si baja
            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Mostrar nombre y confianza
            label = f"{identity} ({confidence*100:.1f}%)"
            cv.putText(frame, label, (x, y-10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        except Exception as e:
            print(f"Error al procesar el rostro: {e}")
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv.putText(frame, "Error", (x, y-10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Mostrar el frame
    cv.imshow("Reconocimiento Facial con FAISS", frame)
    
    # Salir con 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv.destroyAllWindows()
cv.destroyAllWindows