"""
PDF Report Builder for PCB Inspection.
Professional layout for computer-vision + RAG-based PCB defect inspection reports.
"""

from collections import Counter
from pathlib import Path
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


BLUE = colors.HexColor("#1F4E79")
DARK = colors.HexColor("#263238")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GRAY = colors.HexColor("#F4F6F7")
MID_GRAY = colors.HexColor("#B0BEC5")
GREEN = colors.HexColor("#2E7D32")
ORANGE = colors.HexColor("#EF6C00")
RED = colors.HexColor("#B71C1C")

PAGE_WIDTH = 21 * cm
LEFT_MARGIN = 1.35 * cm
RIGHT_MARGIN = 1.35 * cm
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

BOX_PADDING = 6
TABLE_PADDING = 4
GAP_WIDTH = 0.35 * cm
HALF_WIDTH = (CONTENT_WIDTH - GAP_WIDTH) / 2
INNER_HALF_WIDTH = HALF_WIDTH - (BOX_PADDING * 2)


def _clean(value, default="-"):
    if value is None or value == "":
        return default
    return str(value)


def _fmt_float(value, digits=3, default="-"):
    try:
        if value is None:
            return default
        return f"{float(value):.{digits}f}"
    except Exception:
        return default


def _safe_paragraph(text: str) -> str:
    return escape(_clean(text)).replace("\n", "<br/>")


def _severity_color(severity: str):
    sev = str(severity or "").upper()
    if sev in {"CRITICAL", "HIGH"}:
        return RED
    if sev == "MEDIUM":
        return ORANGE
    if sev == "LOW":
        return GREEN
    return DARK


def _status_from_response(rag_response: dict) -> str:
    status = rag_response.get("acceptability_status")
    if status:
        return str(status).replace("_", " ").upper()

    action = rag_response.get("recommended_action")
    if action:
        return str(action).replace("_", " ").upper()

    return "REVIEW"


def _action_from_response(rag_response: dict) -> str:
    action = rag_response.get("recommended_action")
    if action:
        return str(action).replace("_", " ").upper()
    return "ENGINEERING REVIEW"


def _average(values):
    clean_values = []
    for value in values:
        try:
            if value is not None:
                clean_values.append(float(value))
        except Exception:
            pass

    if not clean_values:
        return None

    return sum(clean_values) / len(clean_values)


def _build_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4

    canvas.setFillColor(BLUE)
    canvas.rect(0, height - 1.05 * cm, width, 1.05 * cm, fill=1, stroke=0)

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(1.4 * cm, height - 0.65 * cm, "AUTOMATED PCB INSPECTION REPORT")

    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(width - 1.4 * cm, height - 0.65 * cm, f"Page {doc.page}")

    canvas.setStrokeColor(MID_GRAY)
    canvas.setLineWidth(0.3)
    canvas.line(1.4 * cm, 1.15 * cm, width - 1.4 * cm, 1.15 * cm)

    canvas.setFillColor(colors.HexColor("#607D8B"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(
        1.4 * cm,
        0.72 * cm,
        "Generated from computer vision output and standards-grounded RAG analysis.",
    )
    canvas.drawRightString(
        width - 1.4 * cm,
        0.72 * cm,
        "Preliminary inspection support document",
    )

    canvas.restoreState()


def _make_styles():
    styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#607D8B"),
            spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "box_title": ParagraphStyle(
            "BoxTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=BLUE,
            spaceBefore=0,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontSize=8.2,
            leading=11.2,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontSize=7,
            leading=9,
            textColor=DARK,
        ),
        "small_white": ParagraphStyle(
            "SmallWhite",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
            textColor=colors.white,
        ),
        "kpi_label": ParagraphStyle(
            "KpiLabel",
            parent=styles["BodyText"],
            fontSize=6.5,
            leading=8,
            textColor=colors.HexColor("#607D8B"),
            alignment=TA_CENTER,
        ),
        "kpi_value": ParagraphStyle(
            "KpiValue",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=DARK,
            alignment=TA_CENTER,
        ),
        "note": ParagraphStyle(
            "Note",
            parent=styles["BodyText"],
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#546E7A"),
            spaceAfter=4,
        ),
    }


def _kpi_table(items, styles):
    row = []

    for label, value in items:
        row.append([
            Paragraph(_safe_paragraph(value), styles["kpi_value"]),
            Paragraph(_safe_paragraph(label), styles["kpi_label"]),
        ])

    table = Table([row], colWidths=[CONTENT_WIDTH / len(items)] * len(items))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.4, MID_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _section_block(title, text, styles):
    if not text:
        return []

    return [
        Paragraph(title, styles["section"]),
        Paragraph(_safe_paragraph(text), styles["body"]),
    ]


def _detections_table(detections, styles, compact=False):
    headers = ["ID", "Defect", "Severity", "Conf.", "Location", "Width", "Height", "Area", "Reference"]

    data = [[Paragraph(h, styles["small_white"]) for h in headers]]

    for idx, d in enumerate(detections, start=1):
        data.append([
            str(idx),
            _clean(d.get("defect_class")),
            _clean(d.get("severity")),
            _fmt_float(d.get("confidence"), 3),
            _clean(d.get("location")),
            _fmt_float(d.get("width_mm"), 3),
            _fmt_float(d.get("height_mm"), 3),
            _fmt_float(d.get("area_mm2"), 2),
            _clean(d.get("reference")),
        ])

    col_widths = [
        0.75 * cm,
        2.75 * cm,
        1.55 * cm,
        1.25 * cm,
        2.15 * cm,
        1.45 * cm,
        1.45 * cm,
        1.65 * cm,
        CONTENT_WIDTH - (0.75 + 2.75 + 1.55 + 1.25 + 2.15 + 1.45 + 1.45 + 1.65) * cm,
    ]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("FONTSIZE", (0, 0), (-1, -1), 6.6 if compact else 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]

    for row_idx, d in enumerate(detections, start=1):
        if row_idx % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#FAFAFA")))
        style_cmds.append(("TEXTCOLOR", (2, row_idx), (2, row_idx), _severity_color(d.get("severity"))))

    table.setStyle(TableStyle(style_cmds))
    return table


def _sources_table(sources, styles):
    if not sources:
        return None

    data = [[
        Paragraph("Source file", styles["small_white"]),
        Paragraph("Chunk", styles["small_white"]),
        Paragraph("Score", styles["small_white"]),
    ]]

    for src in sources[:8]:
        data.append([
            Paragraph(_safe_paragraph(src.get("source_file")), styles["small"]),
            _clean(src.get("chunk_index")),
            _fmt_float(src.get("score"), 3),
        ])

    table = Table(data, colWidths=[CONTENT_WIDTH - 4.0 * cm, 2.0 * cm, 2.0 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _location_table(detections, styles, width):
    if not detections:
        return None

    counts = Counter(d.get("location", "unknown") for d in detections)

    zones = [
        "top-left", "top-center", "top-right",
        "middle-left", "middle-center", "middle-right",
        "bottom-left", "bottom-center", "bottom-right",
    ]

    data = []
    for row in range(3):
        cells = []
        for col in range(3):
            zone = zones[row * 3 + col]
            cells.append(Paragraph(
                f"<b>{counts.get(zone, 0)}</b><br/>{zone.replace('-', ' ').title()}",
                styles["small"],
            ))
        data.append(cells)

    table = Table(data, colWidths=[width / 3] * 3)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.35, MID_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _largest_features_table(detections, styles, width, limit=5):
    sortable = []
    for idx, d in enumerate(detections, start=1):
        try:
            area = float(d.get("area_mm2"))
        except Exception:
            area = None
        if area is not None:
            sortable.append((idx, d, area))

    if not sortable:
        return None

    sortable.sort(key=lambda x: x[2], reverse=True)

    data = [[
        Paragraph("ID", styles["small_white"]),
        Paragraph("Location", styles["small_white"]),
        Paragraph("Area", styles["small_white"]),
    ]]

    for idx, d, area in sortable[:limit]:
        data.append([
            f"D{idx}",
            _clean(d.get("location")),
            _fmt_float(area, 2),
        ])

    table = Table(data, colWidths=[width * 0.22, width * 0.48, width * 0.30])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), TABLE_PADDING),
        ("RIGHTPADDING", (0, 0), (-1, -1), TABLE_PADDING),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _metadata_table(metadata, styles):
    if not metadata:
        return None

    fields = [
        ("LLM", metadata.get("model")),
        ("Embeddings", metadata.get("embedding_model")),
        ("Collection", metadata.get("qdrant_collection")),
        ("Top-K", metadata.get("top_k")),
        ("Threshold", metadata.get("score_threshold")),
        ("Latency ms", _fmt_float(metadata.get("latency_ms"), 2)),
    ]

    data = []
    for label, value in fields:
        if value not in [None, ""]:
            data.append([
                Paragraph(f"<b>{label}</b>", styles["small"]),
                Paragraph(_safe_paragraph(value), styles["small"]),
            ])

    if not data:
        return None

    table = Table(data, colWidths=[4.0 * cm, CONTENT_WIDTH - 4.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build_pdf_report(
    rag_response: dict,
    inspection_payload: dict,
    annotated_image_path: str,
    output_path: str,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=RIGHT_MARGIN,
        leftMargin=LEFT_MARGIN,
        topMargin=1.45 * cm,
        bottomMargin=1.35 * cm,
    )

    styles = _make_styles()
    story = []

    detections = inspection_payload.get("detections", []) or []
    report = rag_response.get("report", {}) or {}
    sources = rag_response.get("sources", []) or []
    metadata = rag_response.get("metadata", {}) or {}

    defect_classes = Counter(d.get("defect_class", "unknown") for d in detections)
    dominant_defect = defect_classes.most_common(1)[0][0] if defect_classes else "-"

    severities = Counter(d.get("severity", "unknown") for d in detections)
    dominant_severity = severities.most_common(1)[0][0] if severities else "-"

    avg_area = _average([d.get("area_mm2") for d in detections])
    avg_conf = _average([d.get("confidence") for d in detections])

    status = _status_from_response(rag_response)
    action = _action_from_response(rag_response)

    story.append(Paragraph("PCB Defect Inspection Report", styles["title"]))
    story.append(Paragraph(
        "Computer vision detection with standards-grounded RAG interpretation",
        styles["subtitle"],
    ))

    summary_data = [
        [
            Paragraph("<b>Report ID</b><br/>Auto-generated", styles["small"]),
            Paragraph(f"<b>Defect class</b><br/>{_safe_paragraph(dominant_defect.replace('_', ' ').title())}", styles["small"]),
            Paragraph(f"<b>Board side</b><br/>{_safe_paragraph(inspection_payload.get('board_side'))}", styles["small"]),
            Paragraph(f"<b>Standard target</b><br/>{_safe_paragraph(inspection_payload.get('standard_target'))}", styles["small"]),
        ],
        [
            Paragraph(f"<b>Acceptability</b><br/>{_safe_paragraph(rag_response.get('acceptability_status'))}", styles["small"]),
            Paragraph(f"<b>Grounding strength</b><br/>{_safe_paragraph(rag_response.get('grounding_strength'))}", styles["small"]),
            Paragraph(f"<b>Product class</b><br/>{_safe_paragraph(inspection_payload.get('product_class'))}", styles["small"]),
            Paragraph(f"<b>Reference hint</b><br/>{_safe_paragraph(detections[0].get('reference') if detections else '-')}", styles["small"]),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[CONTENT_WIDTH / 4] * 4)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    kpis = [
        ("Detected instances", str(len(detections))),
        ("Avg area mm²", _fmt_float(avg_area, 2)),
        ("Avg confidence", _fmt_float(avg_conf, 2)),
        ("Dominant severity", dominant_severity),
    ]
    story.append(_kpi_table(kpis, styles))
    story.append(Spacer(1, 10))

    if annotated_image_path and Path(annotated_image_path).exists():
        story.append(Paragraph("Annotated PCB Image", styles["section"]))
        story.append(Image(annotated_image_path, width=CONTENT_WIDTH, height=8.7 * cm))
        story.append(Spacer(1, 8))

    story += _section_block("Executive Summary", report.get("detection_summary"), styles)
    story += _section_block("Standards-Based Interpretation", report.get("standards_interpretation"), styles)

    risk = report.get("technical_risk")
    recommendation = report.get("recommendation")

    if risk or recommendation:
        left_width = (CONTENT_WIDTH - 0.4 * cm) / 2
        right_width = left_width

        left_box = [
            Paragraph("Technical Risk / Implication", styles["box_title"]),
            Spacer(1, 4),
            Paragraph(_safe_paragraph(risk), styles["body"]) if risk else Paragraph("-", styles["body"]),
        ]

        right_box = [
            Paragraph("Preliminary Disposition / Recommendation", styles["box_title"]),
            Spacer(1, 4),
            Paragraph(_safe_paragraph(recommendation), styles["body"]) if recommendation else Paragraph("-", styles["body"]),
        ]

        two_col = Table([[left_box, right_box]], colWidths=[left_width, right_width])
        two_col.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOX", (0, 0), (-1, -1), 0.35, MID_GRAY),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, MID_GRAY),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(two_col)
        story.append(Spacer(1, 8))

    story += _section_block("Grounding Note / Limitations", report.get("grounding_disclaimer"), styles)

    story.append(PageBreak())

    story.append(Paragraph("Detection Inventory", styles["title"]))
    story.append(Paragraph(
        "Detailed values preserved from the API payload. Numeric fields are rounded only where needed for readability.",
        styles["subtitle"],
    ))

    if detections:
        story.append(_detections_table(detections, styles, compact=len(detections) > 12))
        story.append(Spacer(1, 10))

    loc_table = _location_table(detections, styles, INNER_HALF_WIDTH)
    largest_table = _largest_features_table(detections, styles, INNER_HALF_WIDTH)

    if loc_table or largest_table:
        story.append(Paragraph("Localization and Feature Size Summary", styles["section"]))

        left_block = [
            Paragraph("<b>Region-Based Localization</b>", styles["small"]),
            Spacer(1, 3),
        ]
        if loc_table:
            left_block.append(loc_table)

        right_block = [
            Paragraph("<b>Largest Detected Features</b>", styles["small"]),
            Spacer(1, 3),
        ]
        if largest_table:
            right_block.append(largest_table)

        summary_grid = Table(
            [[left_block, "", right_block]],
            colWidths=[HALF_WIDTH, GAP_WIDTH, HALF_WIDTH],
        )
        summary_grid.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("BOX", (0, 0), (0, 0), 0.35, MID_GRAY),
            ("BOX", (2, 0), (2, 0), 0.35, MID_GRAY),

            ("LEFTPADDING", (0, 0), (0, 0), BOX_PADDING),
            ("RIGHTPADDING", (0, 0), (0, 0), BOX_PADDING),
            ("TOPPADDING", (0, 0), (0, 0), BOX_PADDING),
            ("BOTTOMPADDING", (0, 0), (0, 0), BOX_PADDING),

            ("LEFTPADDING", (2, 0), (2, 0), BOX_PADDING),
            ("RIGHTPADDING", (2, 0), (2, 0), BOX_PADDING),
            ("TOPPADDING", (2, 0), (2, 0), BOX_PADDING),
            ("BOTTOMPADDING", (2, 0), (2, 0), BOX_PADDING),

            ("LEFTPADDING", (1, 0), (1, 0), 0),
            ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ("TOPPADDING", (1, 0), (1, 0), 0),
            ("BOTTOMPADDING", (1, 0), (1, 0), 0),
        ]))
        story.append(summary_grid)
        story.append(Spacer(1, 10))

    sources_table = _sources_table(sources, styles)
    if sources_table:
        story.append(Paragraph("Retrieved Sources", styles["section"]))
        story.append(sources_table)
        story.append(Spacer(1, 10))

    metadata_table = _metadata_table(metadata, styles)
    if metadata_table:
        story.append(Paragraph("Inference Stack", styles["section"]))
        story.append(metadata_table)
        story.append(Spacer(1, 10))

    queries = metadata.get("report_retrieval_queries") or []
    if queries:
        story.append(Paragraph("Retrieval Query Trace", styles["section"]))
        for idx, query in enumerate(queries, start=1):
            story.append(Paragraph(f"{idx}. {_safe_paragraph(query)}", styles["note"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is automatically generated and should be reviewed by qualified "
        "inspection personnel before final disposition. RAG outputs are preliminary interpretations and do "
        "not replace formal acceptance criteria, procurement documentation, or expert engineering review.",
        styles["note"],
    ))

    doc.build(story, onFirstPage=_build_header_footer, onLaterPages=_build_header_footer)
    return output_path
