# =============================================================================
# app.py — PCB Defect Inspector · Streamlit UI
# =============================================================================

import json
import streamlit as st

import config
from inference import (
    ensure_model_exists,
    load_model,
    run_inference,
    get_annotated_image,
    save_annotated_image
)
from rag_utils import build_rag_prompt
from utils import save_uploaded_image, save_payload_json, save_rag_prompt, generate_pcb_id
from payload_builder import (
    enrich_yolo_results,
    build_endpoint_payload,
)
from rag_api_client import send_to_rag_api
from report_pdf import build_pdf_report
import fitz

def render_pdf_preview(pdf_path: str, zoom: float = 1.8):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if "pdf_page_index" not in st.session_state:
        st.session_state["pdf_page_index"] = 0

    page_index = st.session_state["pdf_page_index"]
    page_index = max(0, min(page_index, total_pages - 1))

    col_prev, col_info, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("⬅️ Anterior", disabled=page_index == 0, key="pdf_prev"):
            st.session_state["pdf_page_index"] -= 1
            st.rerun()

    with col_info:
        st.markdown(
            f"<div style='text-align:center; font-weight:600;'>Página {page_index + 1} de {total_pages}</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        if st.button("Siguiente ➡️", disabled=page_index >= total_pages - 1, key="pdf_next"):
            st.session_state["pdf_page_index"] += 1
            st.rerun()

    page = doc[page_index]
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img_bytes = pix.tobytes("png")

    st.image(
        img_bytes,
        caption=f"Vista previa — Página {page_index + 1}",
        width="stretch",
    )

    doc.close()

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Defect Inspector",
    page_icon="🔬",
    layout="wide",
)

# ─── Ajuste 3: verificar / descargar modelo antes de cargarlo ─────────────────
try:
    ensure_model_exists()
except (FileNotFoundError, RuntimeError, ImportError) as e:
    st.error(f"⚠️ Modelo no disponible:\n\n{e}")
    st.stop()

# ─── Carga del modelo (cacheado: se carga una sola vez por sesión) ────────────
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
        help="1 = Consumo general | 2 = Industrial | 3 = Alta confiabilidad",
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

    st.markdown("---")
    st.subheader("📐 Dimensiones físicas de la PCB")

    pcb_dims_known = st.checkbox(
        "¿Conoces las dimensiones reales?",
        value=config.PCB_DIMENSIONS_KNOWN,
        help="Si activas esto, el sistema convertirá píxeles a milímetros.",
    )

    if pcb_dims_known:
        # Ajuste 1: pcb_w y pcb_h se pasan directamente a enrich_yolo_results
        pcb_w = st.number_input("Ancho PCB (mm)", value=config.PCB_WIDTH_MM, step=1.0, min_value=1.0)
        pcb_h = st.number_input("Alto PCB (mm)",  value=config.PCB_HEIGHT_MM, step=1.0, min_value=1.0)
    else:
        pcb_w = config.PCB_WIDTH_MM   # valor por default, no se usará en cálculos
        pcb_h = config.PCB_HEIGHT_MM

    st.markdown("---")
    st.subheader("🌐 API RAG")

    rag_api_url = config.get_api_url("report_generate")

    if rag_api_url:
        st.success("API RAG configurada")
    else:
        st.error("API RAG no configurada. Define RAG_API_BASE_URL en secrets.toml.")

    board_side = st.selectbox(
        "Cara de la PCB",
        options=["top", "bottom"],
        index=0,
    )

    product_class = st.text_input(
        "Clase del producto",
        value="unknown",
        help="Identificador del tipo de producto que inspecciona el equipo.",
    )

    st.markdown("---")
    st.caption(f"Modelo: `{config.MODEL_PATH}`")
    st.caption(f"Dispositivo: `{'GPU' if device == 0 else 'CPU'}`")


# ─── Título principal ─────────────────────────────────────────────────────────
st.title("🔬 PCB Defect Inspector")
st.markdown(
    "Sube una imagen de PCB para detectar defectos, "
    "generar el payload enriquecido y enviarlo al endpoint del equipo."
)

# ─── Upload de imagen ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Sube una imagen PCB (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("👆 Sube una imagen para comenzar la inspección.")
    st.stop()

process_image = st.button("🔍 Procesar imagen", type="primary")

if process_image:
    st.session_state["has_processed"] = True
else:
    if not st.session_state.get("has_processed", False):
        st.stop()

# ─── Procesamiento ────────────────────────────────────────────────────────────
if process_image:
    with st.spinner("Procesando imagen..."):

        # 1. Guardar imagen subida en disco
        image_path = save_uploaded_image(uploaded_file)
        pcb_id     = generate_pcb_id(uploaded_file.name)

        # 2. Inferencia YOLO
        results = run_inference(
            model, device, image_path,
            conf=conf_threshold,
            iou=iou_threshold,
        )

        # 3. Imagen anotada (RGB para st.image + BGR guardada en disco)
        annotated_rgb  = get_annotated_image(results)
        annotated_path = save_annotated_image(results, pcb_id)

        # 4. Payload interno enriquecido
        # Ajuste 1: pcb_w y pcb_h de la UI llegan correctamente aquí
        payload = enrich_yolo_results(
            results,
            class_names=class_names,
            image_id=pcb_id,
            pcb_dimensions_known=pcb_dims_known,
            pcb_width_mm=pcb_w,
            pcb_height_mm=pcb_h,
            annotated_image_path=annotated_path,
        )

        # 5. Prompt RAG
        rag_prompt = build_rag_prompt(payload, ipc_class=ipc_class)

        # 6. Guardar en disco
        save_payload_json(payload, pcb_id)
        save_rag_prompt(rag_prompt, pcb_id)

        # 7. Payload para el endpoint
        inspection_payload = build_endpoint_payload(
            enriched_payload=payload,
            annotated_image_path=annotated_path,
            product_class=product_class,
            board_side=board_side,
        )

        st.session_state["pcb_id"] = pcb_id
        st.session_state["payload"] = payload
        st.session_state["rag_prompt"] = rag_prompt
        st.session_state["inspection_payload"] = inspection_payload
        st.session_state["annotated_rgb"] = annotated_rgb
        st.session_state["annotated_path"] = annotated_path

if not st.session_state.get("has_processed", False):
    st.stop()

pcb_id = st.session_state["pcb_id"]
payload = st.session_state["payload"]
rag_prompt = st.session_state["rag_prompt"]
inspection_payload = st.session_state["inspection_payload"]
annotated_rgb = st.session_state["annotated_rgb"]
annotated_path = st.session_state["annotated_path"]

# ─── Resultados ───────────────────────────────────────────────────────────────
status = payload["overall_status"]
STATUS_COLORS = {
    "REJECT":            "🔴",
    "REVIEW":            "🟡",
    "ACCEPT_WITH_NOTES": "🟠",
    "ACCEPT":            "🟢",
}
icon = STATUS_COLORS.get(status, "⚪")

st.markdown(f"## {icon} Resultado: `{status}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total defectos",  payload["total_defects"])
col2.metric("Críticos",        payload["critical_defects"])
col3.metric("Sistémico",       "Sí" if payload["is_systemic"] else "No")
col4.metric("IPC Class",       ipc_class)

st.markdown("---")

# ── Imagen anotada ────────────────────────────────────────────────────────────
st.subheader("🖼️ Imagen anotada")
st.image(annotated_rgb, caption=f"PCB ID: {pcb_id}", width="stretch")

st.markdown("---")

# ── Defectos detectados ───────────────────────────────────────────────────────
if payload["defects"]:
    st.subheader(f"📋 Defectos detectados ({payload['total_defects']})")

    for d in payload["defects"]:
        severity  = d["severity"]
        sev_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")

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

# ── Payload JSON interno ──────────────────────────────────────────────────────
with st.expander("📦 Payload interno (enriquecido)"):
    st.json(payload)
    st.download_button(
        label="⬇️ Descargar JSON interno",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=f"{pcb_id}_payload.json",
        mime="application/json",
    )

# ── Payload del endpoint ──────────────────────────────────────────
with st.expander("📤 Payload del endpoint (formato del equipo)"):
    st.json(inspection_payload)
    st.download_button(
        label="⬇️ Descargar JSON endpoint",
        data=json.dumps(inspection_payload, indent=2, ensure_ascii=False),
        file_name=f"{pcb_id}_inspection_payload.json",
        mime="application/json",
    )

st.markdown("---")

# ── Ajuste 4: Envío al endpoint ───────────────────────────────────────────────
st.subheader("📄 Generar reporte técnico con RAG")

if not rag_api_url:
    st.error("No se puede generar el reporte porque RAG_API_URL no está configurado.")
else:
    user_question = st.text_input(
        "Pregunta para el RAG (opcional)",
        placeholder="¿Qué acción correctiva recomiendas para los defectos críticos?",
    )

    if st.button("📡 Generar reporte RAG", type="primary"):

        final_inspection_payload = build_endpoint_payload(
            enriched_payload=payload,
            annotated_image_path=annotated_path,
            standard_target="IPC-A-600",
            product_class=product_class,
            board_side=board_side,
            user_question=user_question if user_question.strip() else None,
        )

        with st.spinner("Consultando API RAG..."):
            result = send_to_rag_api(
                payload=final_inspection_payload,
                rag_url=rag_api_url,
                timeout=config.RAG_API_TIMEOUT,
            )

        if result["success"]:
            st.success(f"✅ Reporte generado correctamente HTTP {result['status_code']}")

            rag_response = result["response"]

            pdf_path = f"{config.REPORTS_DIR}/{pcb_id}_inspection_report.pdf"

            build_pdf_report(
                rag_response=rag_response,
                inspection_payload=final_inspection_payload,
                annotated_image_path=annotated_path,
                output_path=pdf_path,
            )

            st.session_state["rag_response"] = rag_response
            st.session_state["pdf_path"] = pdf_path
        else:
            st.error(f"❌ Error al generar reporte, Status code: {result.get('status_code')}")
            st.error(f"Detalle: {result.get('error')}")

    if st.session_state.get("pdf_path"):
        st.subheader("📑 Vista previa del reporte PDF")
        render_pdf_preview(st.session_state["pdf_path"], zoom=1.8)

    if st.session_state.get("pdf_path"):
        with open(st.session_state["pdf_path"], "rb") as pdf_file:
            st.download_button(
                label="⬇️ Descargar reporte PDF",
                data=pdf_file,
                file_name=f"{pcb_id}_inspection_report.pdf",
                mime="application/pdf",
                key=f"download_pdf_{pcb_id}",
            )

st.markdown("---")
st.caption(f"Inspection ID: `{pcb_id}` | Timestamp: {payload['timestamp']}")
