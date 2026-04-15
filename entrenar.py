import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
import pickle
import os

# Ruta correcta del dataset (seg_train)
PATH_DATA = r"C:\Users\ariel\OneDrive\Documentos\IFTS\Nueva carpeta\archive\seg_train\seg_train"

# Rutas de guardado
PATH_MODELO = r"C:\Users\ariel\OneDrive\Documentos\IFTS\Nueva carpeta\mi_modelo.keras"
PATH_CLASES = r"C:\Users\ariel\OneDrive\Documentos\IFTS\Nueva carpeta\clases.pkl"

IMG_SIZE = (299, 299)  # Usamos Xception que funciona mejor con 299x299 en vez de 255
BATCH_SIZE = 32 # Cantidad de imágenes por lote

if not os.path.exists(PATH_DATA):
    print(f"ERROR: No se encuentra la carpeta en: {PATH_DATA}")
    print("Revisar si se escribió bien el path de las carpetas.")
else:
    # Carga el dataset detectando las subcarpetas (buildings, forest, glacier, mountain, sea, street)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_DATA, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical' # Divide el dataset en entrenamiento y validación. El 20% de las imágenes se usarán para validación.
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_DATA, validation_split=0.2, subset="validation", seed=123, 
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )

    # Guarda las 6 clases reales
    class_names = train_ds.class_names
    print(f"Clases detectadas: {class_names}") # Imprime las clases detectadas.
    
    with open(PATH_CLASES, 'wb') as f: # Abre el archivo en modo escritura binaria.
        pickle.dump(class_names, f) # Guarda las clases en el archivo.

    # Aumento de datos (Data Augmentation) para evitar sobreajuste
    data_augmentation = tf.keras.Sequential([ # Aumenta la cantidad de datos de entrenamiento.
        layers.RandomFlip("horizontal"), # Gira la imagen horizontalmente como un espejo.
        layers.RandomRotation(0.2), # Gira la imagen hasta un 20%.
        layers.RandomZoom(0.2), # Acerca o aleja la imagen.
    ])

    # Configurar modelo Xception
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)) # Carga el modelo Xception pre-entrenado en ImageNet. ImagenNet es un dataset con millones de imagenes de todo el mundo.
    base_model.trainable = False # No entrena el modelo Xception. Lo deja tal cual viene de fabrica. No lo reentrena.

    model = models.Sequential([
        tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)), # Define el tamaño de la imagen de entrada.
        data_augmentation,
        layers.Rescaling(1./127.5, offset=-1), # Preprocesamiento crucial para Xception (escala a [-1, 1])
        base_model, 
        layers.GlobalAveragePooling2D(), # Convierte el mapa de características 2D en un vector 1D.
        layers.Dense(512, activation='relu'), # Capa densa con activación ReLU.
        layers.Dropout(0.4), # Dropout para evitar memorizar. Apaga el 40% de las neuronas aleatoriamente.
        layers.Dense(len(class_names), activation='softmax') # Capa de salida con activación softmax. Pasa los valores a porcentajes.
    ])

    # Configurar el optimizador con un learning rate pequeño
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping para detenerse si no mejora y Callback para guardar el mejor modelo
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True), # Detiene el entrenamiento si no mejora
        tf.keras.callbacks.ModelCheckpoint(PATH_MODELO, monitor='val_accuracy', save_best_only=True) # Guarda el mejor modelo
    ]

    print(" Iniciando entrenamiento final. Está procesando mas de 14.000 imágenes.")
    # Aumentar las épocas a 15, el EarlyStopping detendrá antes si es necesario
    history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks) #  Entrena el modelo.

    # Guardar
    model.save(PATH_MODELO)
    print(f"⭐ PROCESO TERMINADO. Mejor modelo guardado en: {PATH_MODELO}")