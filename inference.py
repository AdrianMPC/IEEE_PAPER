# =============================================================================
# inference.py — Carga del modelo YOLO e inferencia sobre imágenes PCB
# =============================================================================

import os
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO

import config


# ─── Ajuste 3: descarga automática del modelo desde Google Drive ──────────────
def ensure_model_exists(
    model_path: str = config.MODEL_PATH,
    gdrive_id:  str = config.MODEL_GDRIVE_ID,
) -> None:
    """
    Verifica que el modelo exista en disco.
    Si no existe y MODEL_GDRIVE_ID está configurado en config.py,
    lo descarga automáticamente con gdown.

    Llama esta función una vez al inicio de app.py, antes de load_model().
    """
    if os.path.exists(model_path):
        return  # ya existe, nada que hacer

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if gdrive_id is None:
        lines = [
            f"Modelo no encontrado en: {model_path}",
            "Opciones:",
            "  1. Copia manualmente el archivo a models/small_11.pt",
            "  2. Configura MODEL_GDRIVE_ID en config.py para descarga automática",
        ]
        raise FileNotFoundError("\n".join(lines))

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown no está instalado. Agrégalo a requirements.txt: gdown>=4.7.3"
        )

    print(f"Modelo no encontrado. Descargando desde Google Drive (ID: {gdrive_id})...")
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, model_path, quiet=False)

    if not os.path.exists(model_path):
        raise RuntimeError(
            "La descarga falló. Verifica que MODEL_GDRIVE_ID sea correcto "
            "y que el archivo sea accesible (público o compartido)."
        )
    print(f"Modelo descargado correctamente en: {model_path}")


# ─── Detección de dispositivo ─────────────────────────────────────────────────
def get_device():
    """Retorna el dispositivo disponible: GPU (0) o CPU."""
    if torch.cuda.is_available():
        return 0
    return "cpu"


# ─── Carga de class names desde data.yaml ────────────────────────────────────
def load_class_names(yaml_path: str = config.DATA_YAML_PATH) -> dict:
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


# ─── Carga del modelo ─────────────────────────────────────────────────────────
def load_model(model_path: str = config.MODEL_PATH):
    """
    Carga el modelo YOLO desde disco.

    Uso recomendado en app.py:
        @st.cache_resource
        def get_model():
            return load_model()

        model, device, class_names = get_model()

    Returns:
        tuple: (model, device, class_names)
    """
    device      = get_device()
    model       = YOLO(model_path)
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
    conf:  float = config.YOLO_CONF,
    iou:   float = config.YOLO_IOU,
    imgsz: int   = config.YOLO_IMGSZ,
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


# ─── Imagen anotada en RGB (para st.image) ───────────────────────────────────
def get_annotated_image(results) -> np.ndarray:
    """
    Genera la imagen con bounding boxes y etiquetas en formato RGB.

    Returns:
        np.ndarray: imagen anotada en RGB lista para st.image()
    """
    annotated_bgr = results[0].plot()
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


# ─── Guardar imagen anotada en disco ─────────────────────────────────────────
def save_annotated_image(
    results,
    pcb_id:     str,
    output_dir: str = config.ANNOTATED_DIR,
) -> str:
    """
    Guarda la imagen anotada en BGR en disco y retorna la ruta del archivo.

    Returns:
        str: ruta absoluta al archivo guardado
    """
    annotated_bgr = results[0].plot()
    out_path = str(Path(output_dir) / f"{pcb_id}_annotated.jpg")
    cv2.imwrite(out_path, annotated_bgr)
    return out_path