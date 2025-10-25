# 🚀 Guía Rápida - Entrenamiento de Reconocimiento Facial

## 📋 Prerequisitos

Asegúrate de tener instaladas las dependencias:

```bash
pip install -r requirements.txt
```

---

## 🎯 Proceso Completo (3 Pasos)

### **Paso 1: Capturar Fotos**

Captura fotos de las personas que quieres entrenar:

```bash
python capture_faces.py --name "Juan"
python capture_faces.py --name "Maria"
python capture_faces.py --name "Pedro"
```

**Instrucciones durante la captura:**
- Mira a la cámara
- Mueve lentamente tu cabeza (diferentes ángulos)
- Se capturarán 30 fotos automáticamente cada 1 segundo
- Presiona `ESPACIO` para captura manual
- Presiona `Q` para salir antes

**Resultado:** Las fotos se guardan en `dataset/NombrePersona/`

---

### **Paso 2: Entrenar el Modelo**

Una vez capturadas las fotos de todas las personas:

```bash
python train_model.py
```

**Qué hace:**
- Lee todas las carpetas en `dataset/`
- Detecta rostros en cada imagen
- Extrae embeddings con FaceNet
- Entrena clasificador SVM
- Guarda el modelo entrenado

**Resultado:**
- `face-embeddings-done-for-classes.npz` (embeddings)
- `face-recognition-model.pkl` (modelo SVM)

---

### **Paso 3: Probar el Reconocimiento**

Ejecuta el sistema en tiempo real:

```bash
python main.py
```

**Controles:**
- `Q`: Salir de la aplicación

**Resultado:** El sistema reconocerá a las personas entrenadas en tiempo real

---

## 📁 Estructura de Carpetas

```
Face-Recognition-with-Facenet/
├── dataset/                    # Carpeta de fotos para entrenamiento
│   ├── Juan/                  # 30 fotos de Juan
│   ├── Maria/                 # 30 fotos de Maria
│   └── Pedro/                 # 30 fotos de Pedro
├── capture_faces.py           # Script para capturar fotos
├── train_model.py             # Script para entrenar modelo
├── main.py                    # Sistema de reconocimiento en vivo
├── face-embeddings-done-for-classes.npz  # Embeddings (generado)
└── face-recognition-model.pkl # Modelo entrenado (generado)
```

---

## 🔧 Opciones Avanzadas

### Capturar más/menos fotos:

```bash
python capture_faces.py --name "Juan" --photos 50
```

### Agregar nuevas personas sin borrar anteriores:

```bash
# 1. Captura nueva persona
python capture_faces.py --name "NuevaPersona"

# 2. Re-entrena (incluye todas las personas en dataset/)
python train_model.py
```

---

## ❓ Solución de Problemas

### El sistema no reconoce a nadie:
- Verifica que entrenaste el modelo después de capturar fotos
- Asegúrate de tener al menos 10-15 fotos por persona
- Verifica que la iluminación sea similar entre entrenamiento y uso

### "No se detecta rostro" durante captura:
- Acércate más a la cámara
- Mejora la iluminación
- Verifica que `haarcascade_frontalface_default.xml` esté en la carpeta

### Error al entrenar:
- Verifica que existe la carpeta `dataset/` con subcarpetas
- Asegúrate de que las fotos contengan rostros claros
- Revisa que todas las dependencias estén instaladas

---

## 🎓 Método Alternativo: Notebook Original

Si prefieres usar el notebook Jupyter original:

1. Abre `model/Face_Recognition_using_Facenet.ipynb` en Google Colab
2. Sube tus fotos manualmente al dataset
3. Ejecuta todas las celdas
4. Descarga los archivos `.pkl` y `.npz` generados

---

## 💡 Tips de Uso

- **Diversidad de ángulos**: Captura fotos mirando al frente, arriba, abajo, izquierda, derecha
- **Iluminación**: Captura con iluminación similar a donde usarás el sistema
- **Calidad**: Usa buena resolución de cámara y enfoque nítido
- **Cantidad**: Mínimo 15 fotos por persona, recomendado 30+

---

¡Listo para entrenar tu sistema de reconocimiento facial! 🎉
