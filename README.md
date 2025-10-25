# Sistema Avanzado de Reconocimiento Facial

## 🚀 Descripción del Proyecto

Este repositorio alberga un sistema avanzado de reconocimiento facial que aprovecha técnicas de aprendizaje profundo de vanguardia y algoritmos de machine learning. El proyecto demuestra capacidades de detección y reconocimiento facial en tiempo real, mostrando experiencia en visión por computadora, deep learning e ingeniería de software.

## 📑 Tabla de Contenidos

- [Características Principales](#características-principales)
- [🎥 Script de Captura Automática de Rostros](#-script-de-captura-automática-de-rostros) 
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Inicio Rápido (Windows)](#-inicio-rápido-windows)
- [Opciones de Entrenamiento](#-opciones-de-entrenamiento)
- [Configuración Avanzada](#️-configuración-avanzada)
- [Solución de Problemas](#-solución-de-problemas)

### Características Principales

- ✅ **Detección Facial MTCNN** - Multi-task Cascaded Convolutional Networks para detección precisa de rostros
- ✅ **Herramienta de Captura Automática** - `capture_faces.py`: Captura 30+ fotos desde webcam automáticamente
- ✅ **Reconocimiento en Tiempo Real** - Detecta y reconoce rostros en transmisión de video en vivo
- ✅ **Embeddings FaceNet** - Representaciones faciales de 512 dimensiones para mayor precisión
- ✅ **Clasificación SVM** - Clasificación facial rápida y eficiente
- ✅ **Integración con Datasets Públicos** - Descarga datasets de CelebA + Kaggle automáticamente
- ✅ **Pipeline de Entrenamiento Automatizado** - `train_model.py`: Entrenamiento con un solo comando
- ✅ **Escalable** - Soporte para 50+ identidades

## 🛠️ Tecnologías Utilizadas

- **Python** - Lenguaje de programación principal
- **OpenCV** - Detección facial y procesamiento de video
- **MTCNN** - Multi-task Cascaded Convolutional Networks para detección facial precisa
- **TensorFlow** - Framework de deep learning
- **Keras-FaceNet** - Modelo FaceNet pre-entrenado para embeddings faciales de 512 dimensiones
- **Scikit-learn** - Clasificador SVM para reconocimiento de identidades
- **NumPy** - Operaciones numéricas y manipulación de arrays
- **Kaggle API** - Descarga automática de datasets públicos (CelebA, VGGFace2)

## 🎥 Script de Captura Automática de Rostros

**`capture_faces.py`** - ¡La forma más fácil de crear tu dataset de entrenamiento!

```bash
# Captura 30 fotos automáticamente desde tu webcam
python capture_faces.py --name "TuNombre"
```

**Características:**
- 📸 **Captura automática** cada 1 segundo (30 fotos en total)
- 🎯 **Detección facial** - Solo captura cuando detecta un rostro
- 📊 **Retroalimentación visual** - Rectángulo verde muestra el rostro detectado
- ⌨️ **Modo manual** - Presiona `ESPACIO` para captura manual
- 💾 **Auto-guardado** - Fotos guardadas en `dataset/TuNombre/`
- ⚙️ **Personalizable** - Ajusta número de fotos: `--photos 50`

**Instrucciones de Uso:**
1. Ejecuta el script con tu nombre
2. Mira a la cámara
3. Rota lentamente tu cabeza (izquierda, derecha, arriba, abajo)
4. Las fotos se capturan automáticamente
5. Presiona `Q` para salir anticipadamente si es necesario

**Ejemplo:**
```bash
# Captura tus fotos
python capture_faces.py --name "Alice"

# Captura fotos de un amigo
python capture_faces.py --name "Bob"

# Captura más fotos para mejor precisión
python capture_faces.py --name "Charlie" --photos 50
```

---

## 📁 Estructura del Proyecto

```
Face-Recognition-with-Facenet/
├── capture_faces.py                  # 📸 Herramienta de captura desde webcam (CARACTERÍSTICA CLAVE)
├── train_model.py                    # 🧠 Pipeline de entrenamiento automatizado
├── download_datasets.py              # 📥 Descargador de datasets públicos
├── main.py                           # 🎥 Sistema de reconocimiento en tiempo real
├── requirements.txt                  # 📦 Dependencias de Python
├── haarcascade_frontalface_default.xml
├── face-embeddings-done-for-classes.npz  # Embeddings entrenados
├── face-recognition-model.pkl            # Modelo SVM entrenado
├── dataset/                          # Imágenes de entrenamiento organizadas por identidad
├── model/
│   └── Face_Recognition_using_Facenet.ipynb  # Notebook original
├── GUIA_RAPIDA.md                    # Guía de inicio rápido (español)
└── SETUP_DATASETS.md                 # Instrucciones de configuración de datasets
```

## 🚀 Inicio Rápido (Windows)

### Paso 1: Instalar Python

**Versión recomendada: Python 3.9 o 3.10**

Usando Winget (recomendado):
```bash
winget install Python.Python.3.10
```

O descarga desde: https://www.python.org/downloads/

### Paso 2: Instalar Dependencias

```bash
# Instalar todos los paquetes necesarios
pip install -r requirements.txt
```

**Dependencias instaladas:**
- opencv-python
- numpy
- tensorflow
- keras-facenet
- scikit-learn
- kaggle

### Paso 3: Configurar Kaggle API (para datasets públicos)

1. Crea una cuenta en https://www.kaggle.com/
2. Ve a Configuración de Cuenta → API → "Create New API Token"
3. Descarga `kaggle.json`
4. Colócalo en: `C:\Users\TU_USUARIO\.kaggle\kaggle.json`
5. Verifica: `kaggle datasets list`

### Paso 4: Descargar Datasets Públicos (Opcional)

**Opción A: Descargar 50 identidades (CelebA + Kaggle Faces)**
```bash
python download_datasets.py
```
- Descarga ~1.5GB
- Toma 15-30 minutos
- Crea 49 carpetas de identidades

**Opción B: Cantidades personalizadas**
```bash
python download_datasets.py --celebs 25 --kaggle 24
```

### Paso 5: Capturar Tu Rostro

```bash
# Captura 30 fotos tuyas
python capture_faces.py --name "TuNombre"
```

**Instrucciones durante la captura:**
- Mira a la cámara y mueve lentamente tu cabeza
- Diferentes ángulos: frente, izquierda, derecha, arriba, abajo
- Presiona `ESPACIO` para captura manual
- Presiona `Q` para salir
- Fotos guardadas en `dataset/TuNombre/`

**Repite para múltiples personas (mínimo 2 requeridas):**
```bash
python capture_faces.py --name "Persona2"
python capture_faces.py --name "Persona3"
```

### Paso 6: Entrenar el Modelo

```bash
python train_model.py
```

**Qué sucede:**
- Lee todas las carpetas en `dataset/`
- Detecta rostros en cada imagen
- Extrae embeddings FaceNet de 512 dimensiones
- Entrena clasificador SVM
- Guarda archivos del modelo:
  - `face-embeddings-done-for-classes.npz`
  - `face-recognition-model.pkl`

**Tiempo de entrenamiento:** 10-20 minutos (depende del tamaño del dataset)

### Paso 7: Ejecutar Reconocimiento en Tiempo Real

```bash
python main.py
```

**Controles:**
- **Q**: Salir de la aplicación

**Salida:** Ventana de webcam con cuadros de detección facial y nombres predichos

## 📊 Opciones de Entrenamiento

### Opción 1: Prueba Rápida (2 personas, webcam local)
```bash
python capture_faces.py --name "Persona1"
python capture_faces.py --name "Persona2"
python train_model.py
python main.py
```

### Opción 2: Solo Datasets Públicos (50+ identidades)
```bash
python download_datasets.py --celebs 25 --kaggle 24
python train_model.py
python main.py
```

### Opción 3: Híbrido (Tu rostro + Datasets públicos)
```bash
# 1. Descargar datasets públicos
python download_datasets.py --celebs 10 --kaggle 10

# 2. Agregar tu rostro
python capture_faces.py --name "TuNombre"

# 3. Entrenar con todas las identidades
python train_model.py

# 4. Probar
python main.py
```

### Opción 4: Método del Notebook Original
```bash
# Abrir en Jupyter o Google Colab
jupyter notebook model/Face_Recognition_using_Facenet.ipynb
# Seguir instrucciones del notebook
```

## 📈 Resultados y Rendimiento

El sistema demuestra alta precisión en reconocimiento facial en tiempo real, identificando exitosamente múltiples individuos en transmisiones de video en vivo.

**Características de rendimiento:**
- **Embeddings FaceNet:** Representaciones faciales de 512 dimensiones
- **Clasificación SVM:** Kernel lineal para balance óptimo velocidad/precisión
- **Procesamiento en tiempo real:** ~20-30 FPS en hardware moderno
- **Rango de detección:** Soporta múltiples rostros simultáneamente
- **Precisión de entrenamiento:** Típicamente >95% con datasets balanceados

## ⚙️ Configuración Avanzada

### Mejorando la Precisión del Reconocimiento

Si el sistema confunde identidades:

1. **Captura más fotos tuyas:**
   ```bash
   python capture_faces.py --name "TuNombre" --photos 50
   ```

2. **Balancea el dataset:** Asegura número similar de fotos por persona

3. **Agrega umbral de confianza** (edita `main.py`):
   ```python
   probabilities = model.predict_proba(ypred)[0]
   max_prob = np.max(probabilities)
   if max_prob > 0.8:  # Umbral de confianza 80%
       face_name = model.predict(ypred)
   else:
       face_name = ["Desconocido"]
   ```

### Personalizando Configuración de Captura

```bash
# Más/menos fotos
python capture_faces.py --name "Persona" --photos 50

# Índice de cámara diferente (si tienes múltiples cámaras)
# Edita capture_faces.py: cv.VideoCapture(1)  # Prueba 0, 1, 2, etc.
```

### Organización del Dataset

Estructura manual del dataset (alternativa al script de captura):
```
dataset/
  ├── NombrePersona1/
  │   ├── imagen1.jpg
  │   ├── imagen2.jpg
  │   └── ...
  └── NombrePersona2/
      ├── imagen1.jpg
      └── ...
```

Luego ejecuta: `python train_model.py`

## ❓ Solución de Problemas

### Problemas de Instalación

**Problema:** Falla la instalación de TensorFlow
```bash
# Prueba la versión solo CPU
pip install tensorflow-cpu
```

**Problema:** Advertencia de versión de scikit-learn
```bash
# Reinstala las dependencias
pip install -r requirements.txt --force-reinstall
```

### Problemas de Ejecución

**Problema:** "No se detectan rostros" durante la captura
- Mejora la iluminación
- Acércate más a la cámara
- Asegura que el rostro esté claramente visible

**Problema:** Persona incorrecta reconocida
- Captura más fotos de la persona correcta
- Balancea el dataset (fotos similares por persona)
- Reduce el número de clases
- Agrega umbral de confianza (ver Configuración Avanzada)

**Problema:** FPS bajo / Procesamiento lento
- Reduce la resolución de cámara en `main.py`:
  ```python
  cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
  ```

**Problema:** Falla la descarga de dataset de Kaggle
- Verifica que `kaggle.json` esté en la ubicación correcta
- Revisa la conexión a internet
- Acepta los términos del dataset en el sitio web de Kaggle

### Problemas de Entrenamiento

**Problema:** Error "Se necesitan al menos 2 clases"
- Captura al menos 2 personas diferentes
- O usa datasets públicos: `python download_datasets.py`

**Problema:** El entrenamiento toma mucho tiempo
- Reduce el número de identidades
- Usa un subconjunto más pequeño de imágenes por persona

## 📚 Documentación Adicional

- **[GUIA_RAPIDA.md](GUIA_RAPIDA.md)** - Guía rápida en español
- **[SETUP_DATASETS.md](SETUP_DATASETS.md)** - Configuración detallada de datasets
- **[datasets_list.md](datasets_list.md)** - Información de datasets públicos disponibles

## Tecnologias

- **FaceNet:** Investigación de reconocimiento facial de Google
- **Keras-FaceNet:** Implementación de modelo pre-entrenado
- **CelebA:** Dataset de atributos faciales a gran escala
- **Kaggle:** Plataforma de hosting de datasets públicos

---

⭐️ Si encuentras útil este proyecto, ¡considera darle una estrella!

**Comandos de Inicio Rápido:**
```bash
# Configuración mínima (2 personas)
pip install -r requirements.txt
python capture_faces.py --name "Persona1"
python capture_faces.py --name "Persona2"
python train_model.py
python main.py

# Configuración completa (50 identidades)
pip install -r requirements.txt
python download_datasets.py
python train_model.py
python main.py
```
