# Clasificador de Paisajes IA - Grupo 6

Este proyecto contiene un sistema de clasificación de imágenes basado en redes neuronales convolucionales e Inteligencia Artificial, entrenado para identificar 6 tipos de paisajes (Edificios, Bosque, Glaciar, Montaña, Mar y Calle). Utiliza Transfer Learning con la arquitectura **Xception** a través de TensorFlow y Keras. Además, cuenta con una interfaz de usuario visual creada con *Shiny for Python*.

## 📂 Archivos del Proyecto

- `entrenar.py`: 
  - Archivo encargado de procesar las imágenes del dataset original para entrenar la inteligencia artificial.
  - Genera lotes (*batch*), realiza aumentación de datos (*Data Augmentation*) para evitar el sobreajuste y entrena el modelo Xception de base ajustando su capa de salida para 6 paisajes. 
  - Utiliza `EarlyStopping` y `ModelCheckpoint` para autoguardar la versión más precisa.
- `app.py`: 
  - Aplicación y servidor web de *Shiny* que carga en memoria el modelo entrenado. 
  - Provee una interfaz estilizada donde los usuarios pueden subir fotos puntuales y obtener como resultado los cinco paisajes más probables y su nivel de confianza (precisión %), mostrando alertas si el origen de la imagen no es identificable claramente.
- `mi_modelo.keras`: El archivo que contiene toda la red neuronal ya entrenada y pre-compilada.
- `clases.pkl`: Archivo binario generado durante el entrenamiento, se utiliza para guardar el diccionario subyacente de etiquetas/clases del procesador y sincronizarlas unívocamente para su traducción al español en el frontend.
- `requirements.txt`: Archivo de entorno con las librerías necesarias.

## 🚀 Instalación y Uso

Dado que el modelo necesita ejecutarse sobre librerías específicas:

**1. Instala las dependencias necesarias** mediante la terminal:
```bash
pip install -r requirements.txt
```

**2. Ejecuta la Aplicación Web** usando Shiny en tu terminal:
```bash
shiny run app.py
```

Esto abrirá la aplicación web en tu navegador de forma local (normalmente en `http://127.0.0.1:8000/`) y estará lista para subir imágenes.
