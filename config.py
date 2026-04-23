# =============================================================================
# config.py — Configuración centralizada del sistema PCB Defect Inspector
# =============================================================================

import os

# ─── Rutas del proyecto ───────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH         = os.path.join(BASE_DIR, "models", "small_11.pt")
DATA_YAML_PATH     = os.path.join(BASE_DIR, "models", "data.yaml")

# ─── Descarga automática del modelo (Ajuste 3) ────────────────────────────────
MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID")   
UPLOADED_IMG_DIR   = os.path.join(BASE_DIR, "uploaded_images")
OUTPUT_DIR         = os.path.join(BASE_DIR, "outputs")
ANNOTATED_DIR      = os.path.join(OUTPUT_DIR, "annotated")
JSON_DIR           = os.path.join(OUTPUT_DIR, "json")

# Crear carpetas si no existen
for _dir in [UPLOADED_IMG_DIR, ANNOTATED_DIR, JSON_DIR]:
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