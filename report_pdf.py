"""
PDF Report Builder for PCB Inspection
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def build_pdf_report(
    rag_response: dict,
    inspection_payload: dict,
    annotated_image_path: str,
    output_path: str,
) -> str:
    """
    Builds a PDF report summarizing the PCB inspection results and RAG analysis.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=14,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = styles["BodyText"]
    story = []

    story.append(Paragraph("PCB Automated Inspection Report", title_style))
    story.append(Paragraph("Computer Vision + Standards-Grounded RAG Analysis", body_style))
    story.append(Spacer(1, 12))

    if annotated_image_path and Path(annotated_image_path).exists():
        story.append(Paragraph("Annotated PCB Image", section_style))
        story.append(Image(annotated_image_path, width=16 * cm, height=10 * cm))
        story.append(Spacer(1, 12))

    detections = inspection_payload.get("detections", [])

    story.append(Paragraph("Detection Summary", section_style))

    table_data = [["Defect", "Severity", "Confidence", "Location", "Reference"]]

    for d in detections:
        table_data.append([
            d.get("defect_class", "-"),
            d.get("severity", "-"),
            str(d.get("confidence", "-")),
            d.get("location", "-"),
            d.get("reference", "-"),
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("RAG Technical Report", section_style))

    report_text = (
        rag_response.get("report")
        or rag_response.get("response")
        or rag_response.get("answer")
        or rag_response.get("content")
        or str(rag_response)
    )

    for paragraph in str(report_text).split("\n"):
        if paragraph.strip():
            story.append(Paragraph(paragraph.strip(), body_style))
            story.append(Spacer(1, 6))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Disclaimer: This report is automatically generated and should be reviewed by qualified" \
        " inspection personnel before final disposition.",
        body_style,
    ))

    doc.build(story)
    return output_path
