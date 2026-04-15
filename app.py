import os

# Optimización de recursos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Desactiva los mensajes de TensorFlow y errores de concurrencia.
os.environ['OMP_NUM_THREADS'] = '1' # Limita el número de hilos, evita errores de concurrencia y sobrecarga del procesador.
os.environ['TF_NUM_INTRAOP_THREADS'] = '1' # Limita el número de hilos
os.environ['TF_NUM_INTEROP_THREADS'] = '1' # Limita el número de hilos

import tensorflow as tf # Importa la librería TensorFlow que sirve para crear modelos de machine learning.
import numpy as np # Importa la librería NumPy que sirve para realizar operaciones matemáticas.
import pickle # Importa la librería Pickle que sirve para guardar y cargar objetos.
from shiny import App, render, ui, reactive # Importa la librería Shiny que sirve para crear aplicaciones web.
from PIL import Image # Importa la librería PIL que sirve para abrir y manipular imágenes.
import pandas as pd # Importa la librería Pandas que sirve para manipular datos.

# Configuración de rutas y traducciones
BASE_DIR = os.path.dirname(__file__) # Obtiene la ruta base del directorio
MODEL_PATH = os.path.join(BASE_DIR, 'mi_modelo.keras') # Ruta del modelo
PKL_PATH = os.path.join(BASE_DIR, 'clases.pkl') # Ruta de las clases

TRADUCCIONES = { 
    "BUILDINGS": "EDIFICIOS", "FOREST": "BOSQUE", "GLACIER": "GLACIAR",
    "MOUNTAIN": "MONTAÑA", "SEA": "MAR", "STREET": "CALLE"
} # Diccionario de traducciones

modelo = None # Inicializa el modelo
class_names = None # Inicializa las clases

def cargar_ia(): # Función para cargar el modelo
    global modelo, class_names
    if modelo is None:
        try:
            modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
            with open(PKL_PATH, 'rb') as f:
                class_names = pickle.load(f)
            return True
        except: return False
    return True

# Interfaz de usuario (UI)
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.markdown("### **Archivo**"),
        # Botón de búsqueda estilizado con CSS
        ui.input_file("input_image", "Cargue la imagen aquí para comenzar el análisis.", 
                      accept=[".jpg", ".jpeg"], 
                      button_label="Buscar...", 
                      placeholder="Sin archivo seleccionado"),
        ui.hr(),
        bg="#f8f9fa"
    ),
    
    ui.head_content(
        ui.tags.style("""
            /* Estilo Global y Fondo */
            body { background-color: #e9ecef; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            
            /* Tarjeta Principal */
            .card { border: none; border-radius: 15px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); margin: 10px; }
            .card-header { background-color: #212529; color: white; font-weight: bold; border-radius: 15px 15px 0 0 !important; padding: 15px; }
            
            /* Caja de Instrucciones */
            .instruction-box { padding: 20px; background-color: #ffffff; border-radius: 10px; border-left: 6px solid #007bff; margin-bottom: 20px; }
            
            /* ESTILO DEL BOTÓN DE SUBIR (Input File) */
            .btn-file { 
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 10px 20px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 6px rgba(0,123,255,0.3) !important;
            }
            .btn-file:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 12px rgba(0,123,255,0.4) !important; }
            .progress-bar { background-color: #007bff !important; }

            /* TABLA DE RESULTADOS - Alineación Crítica */
            .table-container { background-color: white; border-radius: 10px; overflow: hidden; border: 1px solid #dee2e6; }
            .custom-table { width: 100%; border-collapse: collapse; margin-bottom: 0; }
            .custom-table th { background-color: #212529; color: white; padding: 12px 20px; text-transform: uppercase; font-size: 0.85em; letter-spacing: 1px; }
            .custom-table td { padding: 15px 20px; border-bottom: 1px solid #eee; vertical-align: middle; }
            
            /* Alineación de columnas */
            .col-paisaje { text-align: left; font-weight: 500; color: #495057; }
            .col-confianza { text-align: right; font-weight: 700; color: #007bff; min-width: 120px; }
            
            /* Imagen */
            .img-output { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 4px solid white; }
        """)
    ),

    ui.card(
        ui.card_header("Sistema de Clasificación de Paisajes - IA"),
        ui.div(
            ui.markdown("#### **Instrucciones:**"),
            ui.markdown("1. **Elija** una foto de un paisaje nítida."),
            ui.markdown("2. **Recorte** la imagen para que el paisaje ocupe la mayor parte."),
            ui.markdown("3. **Suba** la imagen usando el menú de la izquierda."),
            class_="instruction-box"
        ),
        ui.layout_column_wrap(
            ui.div(
                ui.markdown("#### **Imagen Cargada:**"),
                ui.output_image("output_image"),
                style="display: flex; flex-direction: column; align-items: center; padding: 10px;"
            ),
            ui.div(
                ui.markdown("#### **Resultado del Análisis:**"),
                ui.output_ui("output_resultado")
            ),
            width=1/2
        )
    ),
    title="Clasificador de Paisajes IA - Grupo 6"
)

# Lógica del servidor (SERVER)
def server(input, output, session): # Función para cargar el modelo
    
    @reactive.calc
    def obtener_predicciones(): # Función para obtener las predicciones
        file = input.input_image() # Obtiene la imagen cargada
        if not file: return None # Si no hay imagen, retorna None
        if not cargar_ia(): return "error" # Si no se puede cargar el modelo, retorna error
        
        img = Image.open(file[0]["datapath"]).convert('RGB').resize((299, 299)) # Convierte la imagen a RGB y la redimensiona
        arr = tf.keras.utils.img_to_array(img) 
        arr = np.expand_dims(arr, 0)

        preds = modelo.predict(arr, verbose=0)[0]
        top_indices = preds.argsort()[-5:][::-1]
        
        resultados = []
        for i in top_indices:
            nombre_en = class_names[i].upper()
            nombre_es = TRADUCCIONES.get(nombre_en, nombre_en)
            resultados.append({
                "Landscape": nombre_es,
                "Prediction": f"{preds[i]*100:.2f} %"
            })
        
        return {"lista": resultados, "top": preds[top_indices[0]]*100, "path": file[0]["datapath"]}

    @render.ui
    def output_resultado():
        res = obtener_predicciones()
        if res is None: return ui.markdown("_Esperando imagen para procesar..._")
        
        warning = ""
        if res["top"] < 30: # Si la confianza es baja (menor al 30%).
            warning = ui.div(
                ui.markdown("⚠️ **Aviso:** No hay suficiente certeza en este paisaje."), # Muestra una advertencia si la confianza es baja
                style="color: #856404; background-color: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffeeba; margin-bottom: 15px;" # Estilo de la advertencia
            )

        # Generar filas con clases CSS para alineación exacta
        filas = [ui.tags.tr(
            ui.tags.td(r["Landscape"], class_="col-paisaje"), # Columna de paisajes
            ui.tags.td(r["Prediction"], class_="col-confianza") # Columna de confianza
        ) for r in res["lista"]] # Itera sobre los resultados y crea las filas

        tabla = ui.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("Paisaje", style="text-align: left;"),
                    ui.tags.th("Confianza", style="text-align: right;")
                )),
                ui.tags.tbody(*filas),
                class_="custom-table"
            ),
            class_="table-container" # Estilo de la tabla.
        )
        
        return ui.div(
            warning, 
            ui.tags.p("Este paisaje es probablemente un:", style="font-weight: 600; font-size: 1.1em; color: #212529; margin-bottom: 12px;"), # Texto que indica que se va a mostrar la tabla.
            tabla # Muestra la tabla.
        )

    @render.image 
    def output_image(): # Función para mostrar la imagen cargada.
        res = obtener_predicciones() # Obtiene las predicciones.
        if res and isinstance(res, dict): # Si las predicciones son válidas.
            return {"src": res["path"], "width": "100%", "class": "img-output", "style": "max-width: 400px;"} # Muestra la imagen cargada.

app = App(app_ui, server)