"""Render HOW_IT_WORKS.md to a clean, readable PDF."""

import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle,
    PageBreak, KeepTogether,
)


SRC = "HOW_IT_WORKS.md"
OUT = "HOW_IT_WORKS.pdf"

INK     = HexColor("#1a1a1a")
MUTED   = HexColor("#555555")
ACCENT  = HexColor("#1f4e79")
RULE    = HexColor("#cccccc")
CODEBG  = HexColor("#f3f4f6")
TBLHEAD = HexColor("#eef2f7")


def make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["Body"] = ParagraphStyle(
        "Body", parent=base["BodyText"],
        fontName="Helvetica", fontSize=11, leading=16,
        textColor=INK, alignment=TA_LEFT, spaceAfter=8,
    )
    styles["H1"] = ParagraphStyle(
        "H1", parent=base["Heading1"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=ACCENT, spaceBefore=8, spaceAfter=14,
    )
    styles["H2"] = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontName="Helvetica-Bold", fontSize=16, leading=20,
        textColor=ACCENT, spaceBefore=18, spaceAfter=8,
    )
    styles["H3"] = ParagraphStyle(
        "H3", parent=base["Heading3"],
        fontName="Helvetica-Bold", fontSize=12.5, leading=16,
        textColor=INK, spaceBefore=12, spaceAfter=4,
    )
    styles["Code"] = ParagraphStyle(
        "Code", parent=base["Code"],
        fontName="Courier", fontSize=9.5, leading=12.5,
        textColor=INK, backColor=CODEBG,
        leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=10,
        borderPadding=6,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet", parent=styles["Body"],
        leftIndent=18, bulletIndent=6, spaceAfter=3,
    )
    styles["Caption"] = ParagraphStyle(
        "Caption", parent=styles["Body"],
        fontSize=9.5, textColor=MUTED, spaceAfter=6,
    )
    return styles


def md_inline(text: str) -> str:
    """Convert a small subset of markdown inline syntax to ReportLab markup."""
    # escape HTML-ish chars first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # bold **x**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # italic *x*  (avoid touching ** that we just did)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    # inline code `x`
    text = re.sub(
        r"`([^`]+)`",
        lambda m: f'<font face="Courier" backColor="#f3f4f6">&nbsp;{m.group(1)}&nbsp;</font>',
        text,
    )
    return text


def parse_table(lines, i):
    """Parse a GFM-style table starting at lines[i]. Returns (rows, next_i)."""
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1
    # Drop the separator row (---|---|...)
    cleaned = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    return cleaned, i


def build_table(rows, styles):
    if not rows:
        return None
    data = [[Paragraph(md_inline(c), styles["Body"]) for c in row] for row in rows]
    n_cols = len(rows[0])
    page_w = A4[0] - 4 * cm
    col_w = page_w / n_cols
    tbl = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TBLHEAD),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl


def build_flowables(md_text, styles):
    flow = []
    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # fenced code blocks
        if stripped.startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code_buf), styles["Code"]))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # headings
        if stripped.startswith("# "):
            flow.append(Paragraph(md_inline(stripped[2:]), styles["H1"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flow.append(Paragraph(md_inline(stripped[3:]), styles["H2"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flow.append(Paragraph(md_inline(stripped[4:]), styles["H3"]))
            i += 1
            continue

        # horizontal rule
        if stripped in ("---", "***", "___"):
            flow.append(Spacer(1, 4))
            flow.append(Table(
                [[""]], colWidths=[A4[0] - 4 * cm], rowHeights=[0.4],
                style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.6, RULE)]),
            ))
            flow.append(Spacer(1, 8))
            i += 1
            continue

        # tables
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"\s*\|[-:\s|]+\|\s*$", lines[i + 1]):
            rows, i = parse_table(lines, i)
            tbl = build_table(rows, styles)
            if tbl is not None:
                flow.append(Spacer(1, 4))
                flow.append(tbl)
                flow.append(Spacer(1, 10))
            continue

        # bullets
        if re.match(r"^\s*[-*]\s+", line):
            indent = len(line) - len(line.lstrip())
            content = re.sub(r"^\s*[-*]\s+", "", line)
            p = Paragraph(md_inline(content), styles["Bullet"], bulletText="•")
            p.style = ParagraphStyle(
                "BulletDyn", parent=styles["Bullet"],
                leftIndent=18 + indent * 6,
                bulletIndent=6 + indent * 6,
            )
            flow.append(p)
            i += 1
            continue

        # numbered lists
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            content = m.group(3)
            num = m.group(2)
            p = Paragraph(md_inline(content), styles["Bullet"], bulletText=f"{num}.")
            flow.append(p)
            i += 1
            continue

        # blank line
        if not stripped:
            flow.append(Spacer(1, 4))
            i += 1
            continue

        # paragraph (collect contiguous non-blank, non-special lines)
        para_lines = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            ns = nxt.strip()
            if (not ns or ns.startswith("#") or ns.startswith("```")
                    or ns.startswith("|") or ns in ("---", "***", "___")
                    or re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^\s*\d+\.\s+", nxt)):
                break
            para_lines.append(nxt)
            i += 1
        text = " ".join(l.strip() for l in para_lines)
        flow.append(Paragraph(md_inline(text), styles["Body"]))

    return flow


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        md_text = f.read()

    styles = make_styles()
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="XAI-Care — How feature importance is computed",
    )
    flow = build_flowables(md_text, styles)
    doc.build(flow)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
