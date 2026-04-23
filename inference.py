# =============================================================================
# inference.py — Carga del modelo YOLO e inferencia sobre imágenes PCB
# =============================================================================

import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO

import config


# ─── Detección de dispositivo ─────────────────────────────────────────────────
def get_device() -> str | int:
    """Retorna el dispositivo disponible: GPU (0) o CPU."""
    if torch.cuda.is_available():
        return 0
    return "cpu"


# ─── Carga de class names desde data.yaml ────────────────────────────────────
def load_class_names(yaml_path: str = config.DATA_YAML_PATH) -> dict[int, str]:
    """
    Carga el mapeo {id: nombre_clase} desde el archivo data.yaml de Roboflow.
    Si el yaml no existe, retorna dict vacío (el modelo usará sus propios names).
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    # Roboflow puede exportarlo como lista o como dict
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


# ─── Carga del modelo (con caché para Streamlit) ──────────────────────────────
def load_model(model_path: str = config.MODEL_PATH):
    """
    Carga el modelo YOLO desde disco.
    
    Uso recomendado en app.py:
        @st.cache_resource
        def get_model():
            return load_model()
    
    Returns:
        tuple: (model, device, class_names)
    """
    device = get_device()
    model  = YOLO(model_path)
    model.to(device)
    class_names = load_class_names()
    # Si el yaml no cargó clases, usar las del propio modelo
    if not class_names:
        class_names = dict(model.names)
    return model, device, class_names


# ─── Inferencia ───────────────────────────────────────────────────────────────
def run_inference(
    model,
    device,
    image_path: str,
    conf: float = config.YOLO_CONF,
    iou:  float = config.YOLO_IOU,
    imgsz: int  = config.YOLO_IMGSZ,
):
    """
    Ejecuta la inferencia YOLO sobre una imagen.

    Args:
        model:      modelo YOLO cargado
        device:     dispositivo (0 para GPU, "cpu" para CPU)
        image_path: ruta absoluta a la imagen
        conf:       umbral de confianza mínima
        iou:        umbral IoU para NMS
        imgsz:      tamaño de imagen de entrada

    Returns:
        list: objetos results de ultralytics
    """
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    return results


# ─── Generación de imagen anotada ─────────────────────────────────────────────
def get_annotated_image(results) -> np.ndarray:
    """
    Genera la imagen con bounding boxes y etiquetas en formato RGB (para st.image).

    Args:
        results: salida de run_inference()

    Returns:
        np.ndarray: imagen anotada en RGB
    """
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb


# ─── Guardado de imagen anotada ───────────────────────────────────────────────
def save_annotated_image(
    results,
    pcb_id: str,
    output_dir: str = config.ANNOTATED_DIR,
) -> str:
    """
    Guarda la imagen anotada en disco y retorna la ruta del archivo.

    Args:
        results:    salida de run_inference()
        pcb_id:     identificador de la PCB (usado en el nombre del archivo)
        output_dir: carpeta donde se guarda la imagen

    Returns:
        str: ruta absoluta al archivo guardado
    """
    annotated_bgr = results[0].plot()
    out_path = str(Path(output_dir) / f"{pcb_id}_annotated.jpg")
    cv2.imwrite(out_path, annotated_bgr)
    return out_path


# ─── Nota sobre @st.cache_resource ───────────────────────────────────────────
# En app.py usa este patrón para evitar recargar el modelo en cada rerun:
#
#   import streamlit as st
#   from inference import load_model
#
#   @st.cache_resource
#   def get_model():
#       return load_model()
#
#   model, device, class_names = get_model()