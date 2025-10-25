import cv2 as cv
import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from keras_facenet import FaceNet
from mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FaceTrainer:
    def __init__(self, dataset_dir='dataset'):
        self.dataset_dir = dataset_dir
        self.target_size = (160, 160)
        self.X = []
        self.y = []
        self.facenet = FaceNet()
        self.detector = MTCNN()
        print("✓ Detector MTCNN inicializado")

    def extract_face(self, filename):
        """Extrae y redimensiona el rostro de una imagen usando MTCNN"""
        img = cv.imread(filename)
        if img is None:
            return None

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Detectar rostros con MTCNN
        try:
            results = self.detector.detect_faces(img)
        except Exception as e:
            return None

        if len(results) == 0:
            return None

        # Obtener el rostro con mayor confianza
        result = results[0]
        x, y, w, h = result['box']
        x, y = abs(x), abs(y)

        # Extraer región del rostro
        face = img[y:y + h, x:x + w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, person_dir):
        """Carga todas las caras de una persona"""
        faces = []
        files = os.listdir(person_dir)

        for img_name in files:
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            path = os.path.join(person_dir, img_name)
            face = self.extract_face(path)

            if face is not None:
                faces.append(face)

        return faces

    def load_dataset(self):
        """Carga todas las personas del dataset"""
        if not os.path.exists(self.dataset_dir):
            print(f"Error: La carpeta '{self.dataset_dir}' no existe")
            print("Crea la carpeta y añade fotos usando: python capture_faces.py --name 'NombrePersona'")
            return False

        subdirs = [d for d in os.listdir(self.dataset_dir)
                   if os.path.isdir(os.path.join(self.dataset_dir, d))]

        if len(subdirs) == 0:
            print(f"Error: No hay carpetas en '{self.dataset_dir}'")
            print("Añade fotos usando: python capture_faces.py --name 'NombrePersona'")
            return False

        print(f"\n=== Cargando dataset de {len(subdirs)} personas ===\n")

        for person_name in subdirs:
            person_path = os.path.join(self.dataset_dir, person_name)
            faces = self.load_faces(person_path)

            if len(faces) > 0:
                print(f"✓ {person_name}: {len(faces)} fotos cargadas")
                self.X.extend(faces)
                self.y.extend([person_name] * len(faces))
            else:
                print(f"✗ {person_name}: No se encontraron rostros válidos")

        if len(self.X) == 0:
            print("\nError: No se cargaron imágenes válidas")
            return False

        print(f"\n✅ Total: {len(self.X)} imágenes de {len(set(self.y))} personas")
        return True

    def extract_embeddings(self):
        """Extrae embeddings usando FaceNet"""
        print("\n=== Extrayendo embeddings con FaceNet ===")
        embeddings = []

        for i, face in enumerate(self.X):
            face = face.astype('float32')
            face = np.expand_dims(face, axis=0)
            embedding = self.facenet.embeddings(face)
            embeddings.append(embedding[0])

            if (i + 1) % 10 == 0:
                print(f"Procesadas: {i + 1}/{len(self.X)}")

        print(f"✅ {len(embeddings)} embeddings extraídos")
        return np.array(embeddings)

    def train_svm(self, embeddings, labels):
        """Entrena el clasificador SVM"""
        print("\n=== Entrenando modelo SVM ===")

        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)

        model = SVC(kernel='linear', probability=True)
        model.fit(embeddings, labels_encoded)

        # Calcular precisión en entrenamiento
        train_acc = model.score(embeddings, labels_encoded)
        print(f"✅ Modelo entrenado - Precisión: {train_acc * 100:.2f}%")

        return model, encoder

    def save_model(self, embeddings, labels, model):
        """Guarda el modelo y embeddings"""
        print("\n=== Guardando modelo ===")

        # Guardar embeddings
        np.savez_compressed('face-embeddings-done-for-classes.npz',
                           embeddings, np.array(labels))
        print("✓ Embeddings guardados: face-embeddings-done-for-classes.npz")

        # Guardar modelo SVM
        with open('face-recognition-model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("✓ Modelo guardado: face-recognition-model.pkl")

        print("\n✅ ¡Entrenamiento completado!")
        print("\nPróximo paso: Ejecuta 'python main.py' para probar el reconocimiento")

    def train(self):
        """Ejecuta el proceso completo de entrenamiento"""
        # Cargar dataset
        if not self.load_dataset():
            return

        # Extraer embeddings
        embeddings = self.extract_embeddings()

        # Entrenar modelo
        model, encoder = self.train_svm(embeddings, self.y)

        # Guardar modelo
        self.save_model(embeddings, self.y, model)

        # Mostrar resumen
        print("\n" + "="*50)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("="*50)
        print(f"Personas entrenadas: {list(set(self.y))}")
        print(f"Total de imágenes: {len(self.X)}")
        print(f"Embeddings por imagen: 512 dimensiones")
        print("="*50)


if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.train()
