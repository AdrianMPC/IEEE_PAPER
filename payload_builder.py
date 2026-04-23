# =============================================================================
# payload_builder.py — Enrichment Layer: YOLO results → JSON payload
# =============================================================================
# Todas las funciones reciben sus dependencias como parámetros (sin globales)
# para ser reutilizables y testeables fuera del entorno de Streamlit.
# =============================================================================

from datetime import datetime
from typing import Optional

import config


# ─── Calibración píxel → milímetro ───────────────────────────────────────────
def make_px_to_mm(
    pcb_width_mm:  float = config.PCB_WIDTH_MM,
    pcb_height_mm: float = config.PCB_HEIGHT_MM,
    image_w_px:    int   = config.IMAGE_W_PX,
    image_h_px:    int   = config.IMAGE_H_PX,
):
    """
    Devuelve una función px_to_mm configurada con la escala de la PCB.
    Se reconstruye cada vez que el usuario cambia las dimensiones en la UI.

        px_to_mm = make_px_to_mm(pcb_width_mm=120.0, pcb_height_mm=90.0)
        w_mm = px_to_mm(w_px, axis='x')
    """
    scale_x = image_w_px / pcb_width_mm
    scale_y = image_h_px / pcb_height_mm

    def px_to_mm(px_value: float, axis: str = "x") -> float:
        scale = scale_x if axis == "x" else scale_y
        return round(px_value / scale, 3)

    return px_to_mm


# ─── Zona de ubicación en grilla 3×3 ─────────────────────────────────────────
def get_location_zone(cx_norm: float, cy_norm: float) -> str:
    """
    Divide la PCB en 9 zonas usando coordenadas normalizadas [0, 1].
    cx_norm, cy_norm provienen del campo xywhn de YOLO.

    Returns:
        str: p.ej. "top-left", "middle-center", "bottom-right"
    """
    col = "left"   if cx_norm < 0.33 else ("center" if cx_norm < 0.66 else "right")
    row = "top"    if cy_norm < 0.33 else ("middle"  if cy_norm < 0.66 else "bottom")
    return f"{row}-{col}"


# ─── Cálculo de severidad operacional ────────────────────────────────────────
def compute_severity(
    defect_name:     str,
    confidence:      float,
    area_mm2:        Optional[float],
    defect_metadata: dict = config.DEFECT_METADATA,
) -> str:
    """
    Severidad final ajustada por:
      - Severidad base del mapeo normativo IPC
      - Confianza del detector (degrada si < 0.50)
      - Área del defecto (escala si > 5 mm²)

    Returns:
        str: "critical" | "high" | "medium" | "low" | "unknown"
    """
    meta     = defect_metadata.get(defect_name, {})
    severity = meta.get("derived_severity", "unknown")

    if confidence < 0.50:
        degradation = {"critical": "high", "high": "medium", "medium": "low"}
        severity = degradation.get(severity, severity)

    if area_mm2 is not None and area_mm2 > 5.0:
        escalation = {"low": "medium", "medium": "high"}
        severity = escalation.get(severity, severity)

    return severity


# ─── Estado global de la PCB ──────────────────────────────────────────────────
def compute_board_status(defects: list) -> str:
    """
    Determina el estado final de la PCB según las severidades detectadas.

    Returns:
        str: "REJECT" | "REVIEW" | "ACCEPT_WITH_NOTES" | "ACCEPT"
    """
    if not defects:
        return "ACCEPT"

    severities = [d["severity"] for d in defects]

    if "critical" in severities:
        return "REJECT"
    if severities.count("high") >= 2:
        return "REVIEW"
    return "ACCEPT_WITH_NOTES"


# ─── Enrichment Layer principal ───────────────────────────────────────────────
def enrich_yolo_results(
    results,
    class_names:          dict,
    defect_metadata:      dict          = config.DEFECT_METADATA,
    image_id:             str           = "PCB_001",
    pcb_dimensions_known: bool          = config.PCB_DIMENSIONS_KNOWN,
    pcb_width_mm:         float         = config.PCB_WIDTH_MM,
    pcb_height_mm:        float         = config.PCB_HEIGHT_MM,
    annotated_image_path: Optional[str] = None,
) -> dict:

    px_to_mm = make_px_to_mm(
        pcb_width_mm=pcb_width_mm,
        pcb_height_mm=pcb_height_mm,
    ) if pcb_dimensions_known else None

    all_defects  = []
    class_counts = {}

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            # ── Datos crudos YOLO ────────────────────────────────────────────
            cls_id     = int(box.cls.item())
            confidence = round(float(box.conf.item()), 4)
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            cx_n, cy_n, w_n, h_n = box.xywhn[0].tolist()

            # ── Dimensiones ──────────────────────────────────────────────────
            w_px    = round(x2 - x1, 1)
            h_px    = round(y2 - y1, 1)
            area_px = round(w_px * h_px, 1)

            w_mm = h_mm = area_mm2 = None
            if pcb_dimensions_known and px_to_mm:
                w_mm     = px_to_mm(w_px, "x")
                h_mm     = px_to_mm(h_px, "y")
                area_mm2 = round(w_mm * h_mm, 4)

            # ── Enriquecimiento semántico ─────────────────────────────────────
            defect_name   = class_names.get(cls_id, f"class_{cls_id}")
            location_zone = get_location_zone(cx_n, cy_n)
            meta          = defect_metadata.get(defect_name, {})
            severity      = compute_severity(defect_name, confidence, area_mm2, defect_metadata)

            class_counts[defect_name] = class_counts.get(defect_name, 0) + 1

            # ── Record del defecto ───────────────────────────────────────────
            defect_record = {
                "defect_id":   f"{image_id}_D{len(all_defects) + 1:03d}",
                "defect_type": defect_name,
                "class_id":    cls_id,
                "confidence":  confidence,
                "severity":    severity,

                # Campos IPC-A-600M
                "ipc_family":                meta.get("ipc_family", "Unknown"),
                "ipc_basis":                 meta.get("ipc_basis", "Nonconforming"),
                "description":               meta.get("description", "No description available"),
                "engineering_justification": meta.get("engineering_justification", "Manual review required"),
                "ipc_reference":             "IPC-A-600",

                "location": {
                    "zone":        location_zone,
                    "bbox_px":     {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center_norm": {"cx": round(cx_n, 4), "cy": round(cy_n, 4)},
                },
                "dimensions": {
                    "width_px":  w_px,
                    "height_px": h_px,
                    "area_px2":  area_px,
                    "width_mm":  w_mm,
                    "height_mm": h_mm,
                    "area_mm2":  area_mm2,
                },
            }
            all_defects.append(defect_record)

    # ── Resumen global ─────────────────────────────────────────────────────────
    critical_count = sum(1 for d in all_defects if d["severity"] == "critical")
    is_systemic    = any(v >= 3 for v in class_counts.values())
    overall_status = compute_board_status(all_defects)

    return {
        "inspection_id":        image_id,
        "timestamp":            datetime.now().isoformat(),
        "total_defects":        len(all_defects),
        "critical_defects":     critical_count,
        "is_systemic":          is_systemic,
        "defects_by_type":      class_counts,
        "overall_status":       overall_status,
        "annotated_image_path": annotated_image_path,
        "defects":              all_defects,
    }

CLASS_NAME_TO_ENDPOINT = {
    "Spurious Copper": "spurious_copper",
    "spurious copper": "spurious_copper",
    "spurious_copper": "spurious_copper",
    "Short": "short",
    "short": "short",
    "Open Circuit": "open_circuit",
    "open circuit": "open_circuit",
    "open_circuit": "open_circuit",
    "Mouse Bite": "mouse_bite",
    "mouse bite": "mouse_bite",
    "mouse_bite": "mouse_bite",
    "Spur": "spur",
    "spur": "spur",
    "Pin Hole": "pin_hole",
    "pin hole": "pin_hole",
    "pin_hole": "pin_hole",
}

def normalize_endpoint_defect_class(name: str) -> str:
    if not name:
        return "unknown"
    return CLASS_NAME_TO_ENDPOINT.get(name.strip(), name.strip().lower().replace(" ", "_"))

def build_endpoint_payload(
    enriched_payload: dict,
    product_class: str = "unknown",
    board_side: str = "top",
    user_question=None,
    standard_target: str = "IPC-A-600",
) -> dict:
    detections = []

    for defect in enriched_payload.get("defects", []):
        location_obj = defect.get("location", {})
        dimensions = defect.get("dimensions", {})

        detections.append({
            "severity": str(defect.get("severity", "UNKNOWN")).upper(),
            "defect_class": normalize_endpoint_defect_class(defect.get("defect_type", "")),
            "confidence": round(float(defect.get("confidence", 0.0)), 3),
            "location": location_obj.get("zone", "unknown"),
            "width_mm": dimensions.get("width_mm"),
            "height_mm": dimensions.get("height_mm"),
            "area_mm2": dimensions.get("area_mm2"),
            "reference": defect.get("ipc_reference", standard_target),
        })

    return {
        "detections": detections,
        "standard_target": standard_target,
        "product_class": product_class,
        "board_side": board_side,
        "user_question": user_question,
    }

def build_delivery_payload(
    inspection_payload: dict,
    annotated_image_path: str | None = None,
) -> dict:
    return {
        "annotated_image_path": annotated_image_path,
        "inspection_payload": inspection_payload,
    }