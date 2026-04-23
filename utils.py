# =============================================================================
# utils.py — Funciones auxiliares: IO, guardado de resultados, batch
# =============================================================================

import json
import os
import glob
import uuid
from pathlib import Path
from typing import Optional

import config


# ─── Guardado de payload JSON ─────────────────────────────────────────────────
def save_payload_json(
    payload:    dict,
    pcb_id:     str,
    output_dir: str = config.JSON_DIR,
) -> str:
    """
    Guarda el payload enriquecido como archivo JSON en disco.

    Args:
        payload:    dict retornado por enrich_yolo_results()
        pcb_id:     identificador de la PCB
        output_dir: carpeta de destino

    Returns:
        str: ruta absoluta al archivo JSON guardado
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json_path = str(Path(output_dir) / f"{pcb_id}_payload.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return json_path


# ─── Guardado de prompt RAG ───────────────────────────────────────────────────
def save_rag_prompt(
    prompt:     str,
    pcb_id:     str,
    output_dir: str = config.JSON_DIR,
) -> str:
    """
    Guarda el prompt RAG como archivo .txt en disco.

    Args:
        prompt:     string generado por build_rag_prompt()
        pcb_id:     identificador de la PCB
        output_dir: carpeta de destino

    Returns:
        str: ruta absoluta al archivo .txt guardado
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    prompt_path = str(Path(output_dir) / f"{pcb_id}_rag_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    return prompt_path


# ─── Guardar imagen subida por el usuario ────────────────────────────────────
def save_uploaded_image(uploaded_file, upload_dir: str = config.UPLOADED_IMG_DIR) -> str:
    """
    Guarda el archivo subido por st.file_uploader en disco.
    Genera un nombre único para evitar colisiones.

    Args:
        uploaded_file: objeto retornado por st.file_uploader()
        upload_dir:    carpeta de destino

    Returns:
        str: ruta absoluta al archivo guardado
    """
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    ext       = Path(uploaded_file.name).suffix
    unique_id = uuid.uuid4().hex[:8]
    filename  = f"{Path(uploaded_file.name).stem}_{unique_id}{ext}"
    file_path = str(Path(upload_dir) / filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# ─── Generar PCB ID único ─────────────────────────────────────────────────────
def generate_pcb_id(filename: Optional[str] = None) -> str:
    """
    Genera un identificador único para la inspección.
    Si se pasa un nombre de archivo, lo usa como base.

    Args:
        filename: nombre original del archivo (sin extensión)

    Returns:
        str: p.ej. "PCB_IMG001_a3f9b2c1"
    """
    unique_suffix = uuid.uuid4().hex[:8].upper()
    if filename:
        base = Path(filename).stem.upper().replace(" ", "_")[:20]
        return f"PCB_{base}_{unique_suffix}"
    return f"PCB_{unique_suffix}"


# ─── Procesar lote de imágenes ───────────────────────────────────────────────
def inspect_batch(
    image_folder: str,
    model,
    device,
    class_names:  dict,
    run_inference_fn,
    enrich_fn,
    build_prompt_fn,
    save_annotated_fn,
    output_dir:   str  = config.OUTPUT_DIR,
    conf:         float = config.YOLO_CONF,
    iou:          float = config.YOLO_IOU,
) -> list[dict]:
    """
    Procesa en batch una carpeta de imágenes PCB.
    Genera un JSON payload y prompt RAG por cada imagen.

    Args:
        image_folder:      carpeta con imágenes .jpg / .png
        model:             modelo YOLO cargado
        device:            dispositivo (GPU o CPU)
        class_names:       dict {int: str}
        run_inference_fn:  función de inference.py
        enrich_fn:         función de payload_builder.py
        build_prompt_fn:   función de rag_utils.py
        save_annotated_fn: función de inference.py
        output_dir:        carpeta base de salida
        conf:              umbral de confianza
        iou:               umbral IoU

    Returns:
        list[dict]: lista de payloads generados
    """
    annotated_dir = str(Path(output_dir) / "annotated")
    json_dir      = str(Path(output_dir) / "json")
    Path(annotated_dir).mkdir(parents=True, exist_ok=True)
    Path(json_dir).mkdir(parents=True, exist_ok=True)

    images = (
        glob.glob(f"{image_folder}/*.jpg")
        + glob.glob(f"{image_folder}/*.jpeg")
        + glob.glob(f"{image_folder}/*.png")
    )

    print(f"🔍 Procesando {len(images)} imágenes desde '{image_folder}'...")
    all_payloads = []

    for img_path in images:
        pcb_id = Path(img_path).stem.upper()

        results       = run_inference_fn(model, device, img_path, conf=conf, iou=iou)
        annotated_path = save_annotated_fn(results, pcb_id, output_dir=annotated_dir)
        payload       = enrich_fn(
            results,
            class_names=class_names,
            image_id=pcb_id,
            annotated_image_path=annotated_path,
        )
        prompt = build_prompt_fn(payload)

        save_payload_json(payload, pcb_id, output_dir=json_dir)
        save_rag_prompt(prompt, pcb_id, output_dir=json_dir)

        all_payloads.append(payload)

        icon = "❌" if payload["overall_status"] == "REJECT" else (
               "⚠️" if payload["overall_status"] == "REVIEW" else "✅")
        print(f"  {icon} {pcb_id}: {payload['overall_status']} | {payload['total_defects']} defectos")

    rejected = sum(1 for p in all_payloads if p["overall_status"] == "REJECT")
    print(f"\n📊 Batch completo: {len(images)} PCBs | {rejected} rechazadas | {len(images) - rejected} OK")

    return all_payloads


# ─── Cargar payload JSON desde disco ─────────────────────────────────────────
def load_payload_json(json_path: str) -> dict:
    """Carga un payload JSON previamente guardado."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Listar reportes existentes ───────────────────────────────────────────────
def list_existing_reports(json_dir: str = config.JSON_DIR) -> list[str]:
    """
    Retorna lista de rutas de JSONs de inspecciones guardadas.
    Útil para mostrar historial en la UI.
    """
    return sorted(glob.glob(f"{json_dir}/*_payload.json"), reverse=True)