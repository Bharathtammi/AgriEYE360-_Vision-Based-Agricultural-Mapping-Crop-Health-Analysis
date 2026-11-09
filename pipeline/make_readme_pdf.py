"""
Turn README.md into a reading PDF (clickable contents + bookmarks + key charts).

  python make_readme_pdf.py
Output: AgriEYE360_project_guide.pdf next to this script.
"""
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "README.md"
OUT = ROOT / "AgriEYE360_project_guide.pdf"
FIGS = [
    ("Disease classifier v2: results per epoch",
     ROOT / "disease_autolabel" / "metrics" / "00_results_grid.png"),
    ("Disease classifier v2: dashboard",
     ROOT / "disease_autolabel" / "metrics" / "dashboard.png"),
    ("Plant (station) positions after the 6 Nov fix",
     ROOT / "agrieye_output" / "station_check" / "stations_map.png"),
    ("Open bolls per plant over time (hand labels + Roboflow boll model)",
     ROOT / "agrieye_output" / "results" / "plots" / "open_boll.png"),
    ("Closed bolls per plant over time",
     ROOT / "agrieye_output" / "results" / "plots" / "closed_boll.png"),
    ("% diseased leaves per plant over time",
     ROOT / "agrieye_output" / "results" / "plots" / "pct_diseased.png"),
    ("Hand labels moved from the panorama onto the 4-POV views (check sheet)",
     ROOT / "agrieye_output" / "pov_dataset" / "check_projection.jpg"),
    ("Roboflow boll model: every mistake on the 355 test images",
     ROOT / "sample_results" / "roboflow_boll_test" / "mistakes.jpg"),
    ("Open (cyan, solid) and closed (magenta, dashed) bolls on the 4 views with the most open bolls",
     ROOT / "sample_results" / "boll_figures" / "00_open_vs_closed.jpg"),
    ("SAM 3 leaf outlines (P09, 31 Jul, east view): every outline is a real leaf, but some leaves have none",
     ROOT / "sample_results" / "sam3_seg_check" / "example_P09_0731_E.jpg"),
    ("One-step boll detector on a test view (cyan = open, magenta = closed, white = hand labels)",
     ROOT / "sample_results" / "roboflow_boll_detect_test" / "preview_P13_2026-08-20_20260820_140906_E.jpg"),
]

F = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Arial", f"{F}\\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", f"{F}\\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", f"{F}\\ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", f"{F}\\arialbi.ttf"))
pdfmetrics.registerFont(TTFont("Consolas", f"{F}\\consola.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")

INK, MUTED, ACCENT, LINE, CODEBG = (colors.HexColor(c) for c in ("#1f2328", "#59636e", "#2a78d6", "#d0d7de", "#f4f6f8"))
S = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=22, leading=27, textColor=INK, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=15, leading=19, textColor=ACCENT, spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=9, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=4, alignment=TA_LEFT),
    "muted": ParagraphStyle("muted", fontName="Arial-Italic", fontSize=9, leading=12, textColor=MUTED, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8, leading=10.5, textColor=INK, wordWrap="CJK"),
    "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8, leading=10.5, textColor=colors.white, wordWrap="CJK"),
    "code": ParagraphStyle("code", fontName="Consolas", fontSize=7.4, leading=9.6, textColor=INK,
                           spaceBefore=0, spaceAfter=0),
    "cap": ParagraphStyle("cap", fontName="Arial-Bold", fontSize=10, leading=13, textColor=INK, spaceBefore=6, spaceAfter=4),
}
STATUS = {"✅": "#1a7f37", "⏳": "#9a6700", "🟡": "#bc4c00", "⬜": "#8c959f"}


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for k, c in STATUS.items():
        t = t.replace(k, f'<font color="{c}">●</font>')
    t = re.sub(r"`([^`]+)`", r'<font name="Consolas" size="8" color="#953800">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", t)
    return t


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                         topMargin=18 * mm, bottomMargin=16 * mm, title="AgriEYE 360 project guide",
                         author="AgriEYE 360")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [frame], onPage=self.deco)])
        self.n = 0

    def deco(self, c, doc):
        c.saveState()
        c.setFont("Arial", 7.5); c.setFillColor(MUTED)
        c.drawString(self.leftMargin, 9 * mm, "AgriEYE 360 · project guide")
        c.drawRightString(A4[0] - self.rightMargin, 9 * mm, f"page {doc.page}")
        c.setStrokeColor(LINE); c.line(self.leftMargin, 12 * mm, A4[0] - self.rightMargin, 12 * mm)
        c.restoreState()

    def beforeDocument(self):
        self.n = 0          # multiBuild runs several passes: keys must repeat identically

    def afterFlowable(self, f):
        lvl = getattr(f, "_lvl", None)
        if lvl is None:
            return
        self.n += 1
        key = f"h{self.n}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(f._plain, key, level=lvl, closed=False)
        self.notify("TOCEntry", (lvl, f._plain, self.page, key))


def heading(text, lvl):
    p = Paragraph(inline(text), S["h1" if lvl == 0 else "h2"])
    p._lvl, p._plain = lvl, re.sub(r"[`*]", "", text)
    return p


def table(rows):
    data = [[Paragraph(inline(c), S["head" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
    n = len(rows[0])
    W = Doc(OUT).width
    widths = {2: [0.42, 0.58], 3: [0.22, 0.39, 0.39], 4: [0.2, 0.15, 0.3, 0.35]}.get(n, [1 / n] * n)
    # path tables: first column holds long paths -> give it more room
    if n == 2 and any("\\" in r[0] for r in rows[1:]):
        widths = [0.55, 0.45]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f3a46")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [t, Spacer(1, 6)]


def parse(md):
    story, lines, i = [], md.splitlines(), 0
    para = []

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), S["body"]))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush(); i += 1; code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            pre = Preformatted("\n".join(code), S["code"], maxLineLength=118, newLineChars="    ")
            box = Table([[pre]], colWidths=[Doc(OUT).width])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODEBG), ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                                     ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
            story += [Spacer(1, 3), box, Spacer(1, 8)]
        elif s.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story += table(rows); continue
        elif s.startswith("# "):
            flush(); story.append(Paragraph(inline(s[2:]), S["title"]))
        elif s.startswith("## "):
            flush(); story.append(heading(s[3:], 0))
        elif s.startswith("### "):
            flush(); story.append(heading(s[4:], 1))
        elif s == "---":
            flush(); story.append(Spacer(1, 4))
        elif re.match(r"^(\s*)(- |\d+\. )", ln):
            flush()
            m = re.match(r"^(\s*)(- |\d+\. )(.*)", ln)
            depth = len(m.group(1)) // 2
            bullet = "•" if m.group(2) == "- " else m.group(2).strip()
            text = m.group(3)
            # continuation lines (indented, not a new bullet)
            while i + 1 < len(lines) and lines[i + 1].startswith(" " * (len(m.group(1)) + 2)) \
                    and not re.match(r"^\s*(- |\d+\. )", lines[i + 1]) and lines[i + 1].strip():
                i += 1; text += " " + lines[i].strip()
            st = ParagraphStyle("li", parent=S["body"], leftIndent=12 + depth * 14, bulletIndent=2 + depth * 14, spaceAfter=2)
            story.append(Paragraph(inline(text), st, bulletText=bullet))
        elif not s:
            flush()
            # a blank line ends a list item; an indented paragraph after a list gets indented too
        elif ln.startswith("  ") and story and isinstance(story[-1], Paragraph) and story[-1].bulletText:
            flush()
            st = ParagraphStyle("lip", parent=S["body"], leftIndent=story[-1].style.leftIndent)
            story.append(Paragraph(inline(s), st))
        elif s.startswith("*") and s.endswith("*") and not s.startswith("**"):
            flush(); story.append(Paragraph(inline(s), S["muted"]))
        else:
            para.append(s)
        i += 1
    flush()
    return story


def main():
    md = SRC.read_text(encoding="utf-8")
    body = parse(md)
    k = next(i for i, f in enumerate(body) if getattr(f, "_lvl", None) == 0)
    front, rest = body[:k], body[k:]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("t0", fontName="Arial-Bold", fontSize=10.5, leading=16, leftIndent=0, textColor=INK),
        ParagraphStyle("t1", fontName="Arial", fontSize=9, leading=13, leftIndent=14, textColor=MUTED),
    ]
    story = front + [
             Paragraph("How to use this PDF: click any line in the contents to jump there; use the "
                       "bookmarks panel in your PDF viewer to go back to any section at any time.", S["body"]),
             Spacer(1, 6), Paragraph("Contents", S["h2"]), toc, PageBreak()] + rest
    figs = [(c, p) for c, p in FIGS if p.exists()]
    if figs:
        story += [PageBreak(), heading("5. Key charts", 0)]
        W = Doc(OUT).width
        for cap, p in figs:
            from PIL import Image as PI
            w, h = PI.open(p).size
            iw = W; ih = iw * h / w
            if ih > 105 * mm:
                ih = 105 * mm; iw = ih * w / h
            story.append(KeepTogether([heading(cap, 1), Image(str(p), iw, ih),
                                       Paragraph(f"File: {p}", S["muted"])]))
    Doc(OUT).multiBuild(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

