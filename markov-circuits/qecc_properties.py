"""
Generate a PDF summary of QECC properties and the role of the weight enumerator.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUT = "markov-circuits/qecc_properties.pdf"

doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontSize=16, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"),
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontSize=10, spaceAfter=16, textColor=colors.HexColor("#555555"),
    alignment=TA_CENTER,
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=12, spaceBefore=16, spaceAfter=6,
    textColor=colors.HexColor("#2c3e50"),
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=9.5, leading=14, spaceAfter=5,
)
cell_style = ParagraphStyle(
    "Cell", parent=styles["Normal"],
    fontSize=8.5, leading=12,
)
cell_bold = ParagraphStyle(
    "CellBold", parent=cell_style,
    fontName="Helvetica-Bold",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def P(text, style=None):
    return Paragraph(text, style or body_style)

def H2(text):
    return Paragraph(text, h2_style)

def rule():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#cccccc"), spaceAfter=8)

# ── Content ───────────────────────────────────────────────────────────────────
story = []

story.append(Paragraph("QECC Properties and the Markov Chain", title_style))
story.append(Paragraph(
    "What poly(n) mixing of the circuit chain enables — and what it does not",
    subtitle_style))
story.append(rule())

# ── Section 1: Table ──────────────────────────────────────────────────────────
story.append(H2("1. QECC Properties: chain vs. no chain"))
story.append(P(
    "The table below classifies key code parameters by whether they are "
    "computable in polynomial time without the Markov chain, and what the "
    "chain contributes in each case."
))
story.append(Spacer(1, 8))

col_w = [3.8*cm, 3.4*cm, 7.4*cm]
header = [
    Paragraph("<b>Property</b>", cell_bold),
    Paragraph("<b>Poly-time\nwithout chain?</b>", cell_bold),
    Paragraph("<b>What the chain adds</b>", cell_bold),
]
rows = [
    ["<b>k</b> — logical qubits",
     "Yes — rank arithmetic",
     "Nothing"],
    ["<b>n</b> — physical qubits",
     "Yes — trivial",
     "Nothing"],
    ["<b>Lower bound on d</b>",
     "Yes — min fundamental\ncircuit size, O(n³)",
     "Nothing"],
    ["<b>Exact d</b>",
     "No — NP-hard\nin general",
     "Samples small circuits; tightens lower bounds on d"],
    ["<b>gap = 0 check</b>",
     "Yes — but costly",
     "Free byproduct: gap = 0 → disconnected chain → d = 2 "
     "(fast rejection of bad codes)"],
    ["<b>Weight enumerator A(z)</b>",
     "No — #P-hard exactly",
     "MCMC approximation of A_w = #{circuits of weight w}, "
     "if chain mixes in poly time"],
    ["<b>Stationary bias L1</b>",
     "No — needs chain",
     "Proxy for code quality: low L1 → circuits spread evenly → good distance profile"],
    ["<b>min|C|/n</b>\n(distance proxy)",
     "Yes — O(n³),\nno chain needed",
     "Nothing"],
    ["<b>Decoding threshold</b>",
     "No — requires\ndecoding simulation",
     "Nothing directly"],
    ["<b>LDPC property</b>",
     "Yes — check row/col\nweights",
     "Nothing"],
]

table_data = [header]
for r in rows:
    table_data.append([Paragraph(r[i], cell_style) for i in range(3)])

tbl = Table(table_data, colWidths=col_w, repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
    ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.HexColor("#f5f5f5"), colors.white]),
    ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",   (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    # Highlight the weight enumerator row
    ("BACKGROUND",   (0, 6), (-1, 6),  colors.HexColor("#fef9e7")),
    ("LINEWIDTH",    (0, 6), (-1, 6),  1.0),
    ("LINECOLOR",    (0, 6), (-1, 6),  colors.HexColor("#f39c12")),
]))
story.append(tbl)

# ── Section 2: Weight enumerator ──────────────────────────────────────────────
story.append(Spacer(1, 12))
story.append(H2("2. Why the Weight Enumerator Matters"))

story.append(P(
    "For a CSS code with X-stabilisers H_X, the <b>weight enumerator</b> of "
    "the circuit space (minimal codewords) is:"
))
story.append(Spacer(1, 4))
story.append(P(
    "<b>A(z) = &sum;<sub>w</sub> A<sub>w</sub> · z<sup>w</sup></b>",
    ParagraphStyle("formula", parent=body_style,
                   fontSize=10.5, leftIndent=1.5*cm, spaceAfter=8)
))
story.append(P(
    "where A<sub>w</sub> = number of circuits (minimal codewords) of weight w."
))
story.append(Spacer(1, 8))

reasons = [
    ("<b>1. Distance is only the first term.</b>",
     "d = min{w : A<sub>w</sub> &gt; 0}. The weight enumerator tells you not "
     "just where circuits start, but how many there are at each weight. A code "
     "with d = 10 but A<sub>10</sub> = 10<sup>6</sup> is far weaker than one "
     "with d = 10 and A<sub>10</sub> = 3."),

    ("<b>2. It governs decoding performance directly.</b>",
     "The block error probability under maximum-likelihood decoding is "
     "approximated by:"
     "<br/><br/>"
     "&nbsp;&nbsp;&nbsp;&nbsp;P<sub>fail</sub> ≈ &sum;<sub>w</sub> "
     "A<sub>w</sub> · f(w, p)"
     "<br/><br/>"
     "where f(w, p) is the probability that a weight-w error pattern is "
     "mistaken for a logical operator. Knowing d alone gives only the leading "
     "term; A(z) gives the full picture."),

    ("<b>3. It distinguishes codes with identical (n, k, d).</b>",
     "Two codes can share the same parameters but have very different A(z) "
     "and therefore very different practical performance. Weight enumerators "
     "are the standard tool for comparing codes within the same family."),

    ("<b>4. The decoding threshold depends on it.</b>",
     "The threshold p* above which decoding fails is sensitive to the "
     "low-weight part of A(z), not just d. A sparse low-weight spectrum "
     "pushes the threshold higher."),
]

for heading, body in reasons:
    story.append(P(heading))
    story.append(P("&nbsp;&nbsp;&nbsp;" + body))
    story.append(Spacer(1, 6))

# ── Section 3: Computational hardness ─────────────────────────────────────────
story.append(H2("3. Why It Is Hard — and Where the Chain Helps"))

story.append(P(
    "Exact computation of A(z) is <b>#P-hard</b>: equivalent to counting "
    "solutions to a system of GF(2) equations. No polynomial-time algorithm "
    "is known or expected."
))
story.append(Spacer(1, 4))
story.append(P(
    "The only practical approach for large codes is <b>MCMC sampling</b>: "
    "run a Markov chain over the circuit space and estimate A<sub>w</sub> "
    "as the fraction of sampled circuits with weight w. This requires the "
    "chain to mix fast — producing an approximately uniform sample in "
    "polynomial time."
))
story.append(Spacer(1, 4))
story.append(P(
    "The <b>poly(n) mixing conjecture</b> for the adjacent circuit chain "
    "on M[H_X] is precisely the claim that this MCMC approach is efficient. "
    "Current empirical evidence (coupling times ~ O(n<sup>α</sup>), "
    "α &lt; 3, across bicycle, toric, and HGP families) is consistent "
    "with the conjecture."
))

doc.build(story)
print(f"Saved → {OUT}")
