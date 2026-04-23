# =============================================================================
# app.py — PCB Defect Inspector · Streamlit UI
# =============================================================================

import json
import streamlit as st

import config
from inference       import load_model, run_inference, get_annotated_image, save_annotated_image
from payload_builder import enrich_yolo_results
from rag_utils       import build_rag_prompt
from utils           import save_uploaded_image, save_payload_json, save_rag_prompt, generate_pcb_id


# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Defect Inspector",
    page_icon="🔬",
    layout="wide",
)

# ─── Carga del modelo (cacheado: se carga una sola vez) ───────────────────────
@st.cache_resource
def get_model():
    with st.spinner("Cargando modelo YOLO..."):
        return load_model()

model, device, class_names = get_model()

# ─── Sidebar — configuración de inspección ───────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuración")

    ipc_class = st.selectbox(
        "Clase IPC del producto",
        options=[1, 2, 3],
        index=config.IPC_CLASS - 1,
        help="1=Consumo general | 2=Industrial | 3=Alta confiabilidad",
    )

    conf_threshold = st.slider(
        "Confianza mínima (conf)",
        min_value=0.10, max_value=0.95,
        value=config.YOLO_CONF, step=0.05,
    )

    iou_threshold = st.slider(
        "IoU (NMS)",
        min_value=0.10, max_value=0.90,
        value=config.YOLO_IOU, step=0.05,
    )

    pcb_dims_known = st.checkbox(
        "¿Conoces las dimensiones físicas de la PCB?",
        value=config.PCB_DIMENSIONS_KNOWN,
    )

    if pcb_dims_known:
        pcb_w = st.number_input("Ancho PCB (mm)", value=config.PCB_WIDTH_MM, step=1.0)
        pcb_h = st.number_input("Alto PCB (mm)",  value=config.PCB_HEIGHT_MM, step=1.0)
    else:
        pcb_w = pcb_h = None

    st.markdown("---")
    st.caption(f"Modelo: `{config.MODEL_PATH}`")
    st.caption(f"Dispositivo: `{'GPU' if device == 0 else 'CPU'}`")


# ─── Título principal ─────────────────────────────────────────────────────────
st.title("🔬 PCB Defect Inspector")
st.markdown(
    "Sube una imagen de PCB para detectar defectos, "
    "generar el payload JSON enriquecido y el prompt para el RAG Chatbot."
)

# ─── Upload de imagen ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Sube una imagen PCB (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("👆 Sube una imagen para comenzar la inspección.")
    st.stop()

# ─── Procesamiento ────────────────────────────────────────────────────────────
with st.spinner("Procesando imagen..."):

    # 1. Guardar imagen subida
    image_path = save_uploaded_image(uploaded_file)
    pcb_id     = generate_pcb_id(uploaded_file.name)

    # 2. Inferencia YOLO
    results = run_inference(
        model, device, image_path,
        conf=conf_threshold,
        iou=iou_threshold,
    )

    # 3. Imagen anotada
    annotated_rgb  = get_annotated_image(results)
    annotated_path = save_annotated_image(results, pcb_id)

    # 4. Payload enriquecido
    # Reconstruir px_to_mm si el usuario cambió las dimensiones
    pcb_config_override = {}
    if pcb_dims_known and pcb_w and pcb_h:
        pcb_config_override = dict(
            pcb_width_mm=pcb_w,
            pcb_height_mm=pcb_h,
        )

    payload = enrich_yolo_results(
        results,
        class_names=class_names,
        image_id=pcb_id,
        pcb_dimensions_known=pcb_dims_known,
        annotated_image_path=annotated_path,
    )

    # 5. Prompt RAG
    rag_prompt = build_rag_prompt(payload, ipc_class=ipc_class)

    # 6. Guardar en disco
    json_path   = save_payload_json(payload, pcb_id)
    prompt_path = save_rag_prompt(rag_prompt, pcb_id)


# ─── Resultados ───────────────────────────────────────────────────────────────
status = payload["overall_status"]
STATUS_COLORS = {
    "REJECT":           "🔴",
    "REVIEW":           "🟡",
    "ACCEPT_WITH_NOTES":"🟠",
    "ACCEPT":           "🟢",
}
icon = STATUS_COLORS.get(status, "⚪")

st.markdown(f"## {icon} Resultado: `{status}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total defectos",   payload["total_defects"])
col2.metric("Críticos",         payload["critical_defects"])
col3.metric("Sistémico",        "Sí" if payload["is_systemic"] else "No")
col4.metric("IPC Class",        ipc_class)

st.markdown("---")

# ── Imagen anotada ────────────────────────────────────────────────────────────
st.subheader("🖼️ Imagen anotada")
st.image(annotated_rgb, caption=f"PCB ID: {pcb_id}", use_container_width=True)

st.markdown("---")

# ── Defectos detectados ───────────────────────────────────────────────────────
if payload["defects"]:
    st.subheader(f"📋 Defectos detectados ({payload['total_defects']})")

    for d in payload["defects"]:
        severity = d["severity"]
        sev_color = {
            "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"
        }.get(severity, "⚪")

        with st.expander(f"{sev_color} `{d['defect_id']}` — {d['defect_type']} ({severity.upper()})"):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Confianza:** {d['confidence'] * 100:.1f}%")
                st.write(f"**Ubicación:** {d['location']['zone']}")
                st.write(f"**Familia IPC:** {d['ipc_family']}")
                st.write(f"**Base IPC:** {d['ipc_basis']}")
            with c2:
                dim = d["dimensions"]
                if dim["width_mm"] is not None:
                    st.write(f"**Tamaño:** {dim['width_mm']} mm × {dim['height_mm']} mm")
                    st.write(f"**Área:** {dim['area_mm2']} mm²")
                else:
                    st.write(f"**Tamaño:** {dim['width_px']} px × {dim['height_px']} px")
            st.write(f"**Descripción:** {d['description']}")
            st.write(f"**Justificación técnica:** {d['engineering_justification']}")
else:
    st.success("✅ No se detectaron defectos con los parámetros actuales.")

st.markdown("---")

# ── Payload JSON ──────────────────────────────────────────────────────────────
with st.expander("📦 Payload JSON completo"):
    st.json(payload)
    st.download_button(
        label="⬇️ Descargar JSON",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=f"{pcb_id}_payload.json",
        mime="application/json",
    )

# ── Prompt RAG ────────────────────────────────────────────────────────────────
with st.expander("🤖 Prompt generado para el RAG Chatbot"):
    st.text_area("Prompt", value=rag_prompt, height=300, label_visibility="collapsed")
    st.download_button(
        label="⬇️ Descargar prompt .txt",
        data=rag_prompt,
        file_name=f"{pcb_id}_rag_prompt.txt",
        mime="text/plain",
    )

st.markdown("---")
st.caption(f"Inspection ID: `{pcb_id}` | Timestamp: {payload['timestamp']}")