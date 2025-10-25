# 📥 Guía de Configuración y Descarga de Datasets

## 🎯 Objetivo

Descargar e integrar **50 identidades** para entrenar el modelo de reconocimiento facial:
- **Lucho**: 1 identidad (tus fotos)
- **CelebA**: 25 celebridades
- **Kaggle Faces**: 24 personas
- **Total**: 50 clases

---

## ⚙️ Configuración Inicial

### **1. Instalar Kaggle CLI**

```bash
pip install kaggle
```

### **2. Configurar Kaggle API**

#### Paso a paso:

1. **Crear cuenta en Kaggle:**
   - Ve a: https://www.kaggle.com/
   - Regístrate o inicia sesión

2. **Obtener API Token:**
   - Ve a tu perfil: https://www.kaggle.com/settings
   - Scroll hasta "API" section
   - Click en "Create New API Token"
   - Se descargará `kaggle.json`

3. **Colocar el token:**

   **Windows:**
   ```bash
   # Crear carpeta .kaggle si no existe
   mkdir C:\Users\TU_USUARIO\.kaggle

   # Mover kaggle.json a esa carpeta
   move C:\Users\TU_USUARIO\Downloads\kaggle.json C:\Users\TU_USUARIO\.kaggle\
   ```

   **Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Verificar configuración:**
   ```bash
   kaggle datasets list
   ```
   Si muestra una lista de datasets, ¡está configurado! ✅

---

## 📥 Descarga de Datasets

### **Opción A: Descarga Automática (Recomendado)**

```bash
python download_datasets.py
```

Esto descargará y organizará automáticamente:
- 25 celebridades de CelebA (~1.3GB)
- 24 personas de Kaggle Faces (~100MB)
- Total: ~1.5GB de descarga

**Tiempo estimado:** 15-30 minutos (según tu conexión)

---

### **Opción B: Personalizar cantidad**

```bash
# Cambiar cantidades
python download_datasets.py --celebs 30 --kaggle 19

# Solo CelebA
python download_datasets.py --celebs 40 --kaggle 0

# Solo Kaggle
python download_datasets.py --celebs 0 --kaggle 49
```

---

### **Opción C: Descarga Manual**

Si prefieres descargar manualmente:

#### CelebA:
1. Ve a: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Click en "Download"
3. Extrae en: `downloads/celeba/`

#### Kaggle Faces:
1. Ve a: https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset
2. Click en "Download"
3. Extrae en: `downloads/kaggle_faces/`

Luego ejecuta:
```bash
python download_datasets.py
```

---

## 🗂️ Estructura Final

Después de la descarga, tendrás:

```
Face-Recognition-with-Facenet/
├── dataset/
│   ├── Lucho/              (30 fotos)
│   ├── Celebrity_01/       (30 fotos)
│   ├── Celebrity_02/       (30 fotos)
│   ├── ...
│   ├── Celebrity_25/       (30 fotos)
│   ├── Person_01/          (30 fotos)
│   ├── Person_02/          (30 fotos)
│   ├── ...
│   └── Person_24/          (30 fotos)
│
├── downloads/              (archivos descargados, puedes borrar después)
│   ├── celeba/
│   └── kaggle_faces/
```

**Total:** 50 carpetas × 30 imágenes = 1,500 imágenes

---

## 🚀 Entrenar el Modelo

Una vez descargados y organizados los datasets:

```bash
python train_model.py
```

**Tiempo estimado de entrenamiento:** 10-20 minutos

---

## ❓ Solución de Problemas

### Error: "kaggle.json not found"
- Verifica que `kaggle.json` esté en: `C:\Users\TU_USUARIO\.kaggle\`
- En Windows, asegúrate de incluir el punto: `.kaggle`

### Error: "403 Forbidden"
- Debes aceptar los términos del dataset en Kaggle
- Ve a la página del dataset y acepta las reglas

### Error: "Not enough images"
- Verifica que las descargas se completaron
- Revisa la carpeta `downloads/` para confirmar

### Descarga muy lenta
- Los datasets son grandes, considera:
  - Reducir cantidad: `--celebs 10 --kaggle 10`
  - Descargar solo uno de los datasets
  - Usar una conexión más rápida

---

## 💾 Espacio en Disco

**Requerimientos:**
- Descargas: ~1.5GB
- Dataset organizado: ~500MB
- Total temporal: ~2GB

**Después de entrenar:**
- Puedes borrar la carpeta `downloads/` para recuperar espacio

---

## 📊 Resumen del Proceso

```
1. Instalar Kaggle CLI → pip install kaggle
2. Configurar API → copiar kaggle.json
3. Descargar datasets → python download_datasets.py
4. Entrenar modelo → python train_model.py
5. Probar sistema → python main.py
```

---

¡Listo para descargar 50 clases de rostros! 🎉
