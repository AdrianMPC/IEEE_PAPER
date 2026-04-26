# =============================================================================
# config.py — Configuración centralizada del sistema PCB Defect Inspector
# =============================================================================

import os
import streamlit as st

# ─── Rutas del proyecto ───────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH         = os.path.join(BASE_DIR, "models", "small_11.pt")
DATA_YAML_PATH     = os.path.join(BASE_DIR, "models", "data.yaml")

# ─── Descarga automática del modelo (Ajuste 3) ────────────────────────────────
MODEL_GDRIVE_ID = st.secrets.get("MODEL_GDRIVE_ID", os.getenv("MODEL_GDRIVE_ID", None))
UPLOADED_IMG_DIR   = os.path.join(BASE_DIR, "uploaded_images")
OUTPUT_DIR         = os.path.join(BASE_DIR, "outputs")
ANNOTATED_DIR      = os.path.join(OUTPUT_DIR, "annotated")
JSON_DIR           = os.path.join(OUTPUT_DIR, "json")

# ─── API RAG ─────────────────────────────────────────────────────────────────
RAG_API_BASE_URL = str(st.secrets.get(
        "RAG_API_BASE_URL",
        os.getenv("RAG_API_BASE_URL", ""
    ))).rstrip("/")
RAG_API_TIMEOUT = int(st.secrets.get("RAG_API_TIMEOUT", os.getenv("RAG_API_TIMEOUT", 90)))

API_V1_PREFIX = "/api/v1"

RAG_ENDPOINTS = {
    "health": f"{API_V1_PREFIX}/health",
    "rag_retrieve": f"{API_V1_PREFIX}/rag/retrieve",
    "rag_query": f"{API_V1_PREFIX}/rag/query",
    "rag_debug": f"{API_V1_PREFIX}/rag/debug",
    "report_generate": f"{API_V1_PREFIX}/report/generate",
    "report_metrics": f"{API_V1_PREFIX}/report/metrics",
    "ragas_dataset": f"{API_V1_PREFIX}/ragas/dataset",
}


def get_api_url(endpoint_name: str) -> str:
    if endpoint_name not in RAG_ENDPOINTS:
        raise ValueError(f"Endpoint no registrado: {endpoint_name}")

    if not RAG_API_BASE_URL:
        return ""

    return f"{RAG_API_BASE_URL}{RAG_ENDPOINTS[endpoint_name]}"

# ─── Reportes PDF ────────────────────────────────────────────────────────────
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Crear carpetas si no existen
for _dir in [UPLOADED_IMG_DIR, ANNOTATED_DIR, JSON_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─── Parámetros de inferencia YOLO ───────────────────────────────────────────
YOLO_CONF      = 0.60   # umbral de confianza
YOLO_IOU       = 0.30   # umbral de IoU para NMS
YOLO_IMGSZ     = 640    # tamaño de imagen de entrada

# ─── Calibración píxel → milímetro ───────────────────────────────────────────
PCB_DIMENSIONS_KNOWN = False
PCB_WIDTH_MM         = 100.0   # ancho real de la PCB en mm
PCB_HEIGHT_MM        = 80.0    # alto real de la PCB en mm
IMAGE_W_PX           = YOLO_IMGSZ
IMAGE_H_PX           = YOLO_IMGSZ

# ─── Clase IPC del producto ───────────────────────────────────────────────────
# 1 = Electrónica de consumo general
# 2 = Electrónica industrial / uso prolongado
# 3 = Alta confiabilidad (aeroespacial, médico, militar)
IPC_CLASS = 2

# ─── Mapeo de Clases Reales del Modelo YOLO ────────────────────────────────────
YOLO_CLASS_NAMES = {
    0: 'Short',
    1: 'Spur',
    2: 'Spurious Copper',
    3: 'Open Circuit',
    4: 'Mouse Bite',
    5: 'Hole Breakout',
    6: 'Conductor Scratch',
    7: 'Conductor Foreign Object',
    8: 'Base Material Foreign Object'
}

# ─── Metadatos normativos IPC-A-600M ─────────────────────────────────────────
DEFECT_METADATA = {
    "Short": {
        "ipc_family":                "Conductive Pattern / Spacing",
        "ipc_basis":                 "Nonconforming",
        "derived_severity":          "critical",
        "description":               "Unintended electrical connection between conductors.",
        "engineering_justification": "Violates minimum electrical spacing and may cause direct functional failure.",
    },
    "Spur": {
        "ipc_family":                "Conductive Pattern / Definition",
        "ipc_basis":                 "Acceptable → Nonconforming",
        "derived_severity":          "medium",
        "description":               "Thin copper protrusion extending from a conductor.",
        "engineering_justification": "May reduce spacing and become critical depending on proximity to adjacent conductors.",
    },
    "Spurious Copper": {
        "ipc_family":                "Conductive Pattern / Residual Metal",
        "ipc_basis":                 "Process Indicator → Nonconforming",
        "derived_severity":          "medium",
        "description":               "Unwanted isolated copper remaining after etching.",
        "engineering_justification": "Risk depends on size and proximity to active conductive features.",
    },
    "Open Circuit": {
        "ipc_family":                "Conductive Pattern / Continuity",
        "ipc_basis":                 "Nonconforming",
        "derived_severity":          "critical",
        "description":               "Break in the intended conductive path.",
        "engineering_justification": "Causes loss of electrical continuity and direct functional failure.",
    },
    "Mouse Bite": {
        "ipc_family":                "Conductive Pattern / Dimensional Integrity",
        "ipc_basis":                 "Nonconforming (threshold-based)",
        "derived_severity":          "high",
        "description":               "Irregular conductor edge defect causing width reduction.",
        "engineering_justification": "May reduce current-carrying capacity and reliability.",
    },
    "Hole Breakout": {
        "ipc_family":                "Holes / Annular Ring Integrity",
        "ipc_basis":                 "Nonconforming (Class-dependent)",
        "derived_severity":          "high",
        "description":               "Hole not fully surrounded by copper annular ring.",
        "engineering_justification": "Weakens mechanical/electrical reliability of plated holes and vias.",
    },
    "Conductor Scratch": {
        "ipc_family":                "Conductive Pattern / Surface Imperfection",
        "ipc_basis":                 "Process Indicator",
        "derived_severity":          "low",
        "description":               "Surface damage affecting the conductor finish or top layer.",
        "engineering_justification": "Often cosmetic unless it significantly reduces thickness or exposes base material.",
    },
    "Conductor Foreign Object": {
        "ipc_family":                "Conductive Pattern / Contamination",
        "ipc_basis":                 "Nonconforming (if conductive)",
        "derived_severity":          "high",
        "description":               "Foreign material present on or near a conductive feature.",
        "engineering_justification": "May induce bridging, corrosion, or unintended conductive paths.",
    },
    "Base Material Foreign Object": {
        "ipc_family":                "Base Material / Subsurface Inclusion",
        "ipc_basis":                 "Acceptable → Nonconforming",
        "derived_severity":          "low",
        "description":               "Foreign inclusion embedded in the base material.",
        "engineering_justification": "Impact depends on size, transparency, and distance to conductive features.",
    },
}

# Atajo: solo severidades base (para compatibilidad si se necesita)
DEFECT_SEVERITY_BASE = {k: v["derived_severity"] for k, v in DEFECT_METADATA.items()}
