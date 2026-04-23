# =============================================================================
# rag_utils.py — Generación del prompt estructurado para el RAG Chatbot
# =============================================================================

import config


def build_rag_prompt(payload: dict, ipc_class: int = config.IPC_CLASS) -> str:
    """
    Convierte el payload enriquecido en un prompt estructurado y optimizado
    para un RAG con base de conocimiento de estándares IPC.

    Args:
        payload:   dict retornado por enrich_yolo_results()
        ipc_class: clase IPC del producto (1, 2 o 3)

    Returns:
        str: prompt listo para enviar al chatbot RAG
    """
    status    = payload["overall_status"]
    total     = payload["total_defects"]
    crits     = payload["critical_defects"]
    systemic  = payload["is_systemic"]

    systemic_note = (
        " Se detectó un patrón sistémico (3 o más instancias del mismo tipo de defecto)."
        if systemic else ""
    )

    # ── Encabezado de inspección ──────────────────────────────────────────────
    header = (
        f"Se realizó una inspección visual automatizada sobre la PCB ID: {payload['inspection_id']}.\n"
        f"Timestamp: {payload['timestamp']}\n"
        f"Resultado general: {status}.\n"
        f"Total de defectos detectados: {total} ({crits} críticos).{systemic_note}\n\n"
    )

    # ── Detalle de cada defecto ───────────────────────────────────────────────
    defect_lines = []
    for d in payload["defects"]:
        dim = d["dimensions"]
        loc = d["location"]

        if dim["width_mm"] is not None:
            size_str = f"{dim['width_mm']} mm × {dim['height_mm']} mm ({dim['area_mm2']} mm²)"
        else:
            size_str = f"{dim['width_px']} px × {dim['height_px']} px"

        line = (
            f"- [{d['severity'].upper()}] {d['defect_type']} "
            f"(ID: {d['defect_id']} | confianza: {d['confidence'] * 100:.1f}%) | "
            f"Familia IPC: {d['ipc_family']} | "
            f"Base de aceptabilidad: {d['ipc_basis']} | "
            f"Ubicación: {loc['zone']} | "
            f"Tamaño: {size_str} | "
            f"Descripción: {d['description']} | "
            f"Justificación técnica: {d['engineering_justification']}"
        )
        defect_lines.append(line)

    if defect_lines:
        defects_block = "Defectos encontrados:\n" + "\n".join(defect_lines)
    else:
        defects_block = "No se detectaron defectos en la inspección."

    # ── Pregunta final al RAG ─────────────────────────────────────────────────
    ipc_refs = list({
        f"{d['ipc_family']} | {d['ipc_basis']}"
        for d in payload["defects"]
    })

    if ipc_refs:
        refs_str = ", ".join(ipc_refs)
        question = (
            f"\n\nBasándote en los estándares {refs_str}, "
            f"por favor genera un reporte de calidad completo que incluya:\n"
            f"(1) Evaluación de cada defecto según criterios IPC.\n"
            f"(2) Evaluación de conformidad según IPC Clase {ipc_class} (definida para este producto).\n"
            f"(3) Recomendación de disposición (aceptar / rechazar / reparar).\n"
            f"(4) Acciones correctivas sugeridas para cada tipo de defecto."
        )
    else:
        question = (
            f"\n\nLa PCB no presentó defectos detectables con los parámetros actuales. "
            f"Confirma conformidad según IPC Clase {ipc_class}."
        )

    return header + defects_block + question


def build_rag_prompt_batch(payloads: list[dict], ipc_class: int = config.IPC_CLASS) -> str:
    """
    Genera un prompt consolidado para un lote de PCBs.
    Útil cuando se procesan múltiples imágenes en una sola sesión.

    Args:
        payloads:  lista de dicts retornados por enrich_yolo_results()
        ipc_class: clase IPC del producto

    Returns:
        str: prompt batch para el RAG
    """
    total_pcbs    = len(payloads)
    rejected      = sum(1 for p in payloads if p["overall_status"] == "REJECT")
    review        = sum(1 for p in payloads if p["overall_status"] == "REVIEW")
    accepted      = total_pcbs - rejected - review
    total_defects = sum(p["total_defects"] for p in payloads)

    header = (
        f"INSPECCIÓN BATCH — {total_pcbs} PCBs procesadas.\n"
        f"Resumen: {rejected} RECHAZADAS | {review} EN REVISIÓN | {accepted} ACEPTADAS.\n"
        f"Total de defectos acumulados: {total_defects}.\n\n"
    )

    sections = []
    for p in payloads:
        sections.append(
            f"PCB {p['inspection_id']}: {p['overall_status']} "
            f"({p['total_defects']} defectos, {p['critical_defects']} críticos)"
        )

    summary = "\n".join(sections)

    question = (
        f"\n\nGenera un análisis de calidad de lote según IPC Clase {ipc_class} que incluya:\n"
        f"(1) Tasa de rechazo y conformidad global.\n"
        f"(2) Defectos sistémicos detectados en múltiples unidades.\n"
        f"(3) Recomendaciones de proceso para reducir la tasa de defectos.\n"
        f"(4) Acciones correctivas prioritarias."
    )

    return header + summary + question