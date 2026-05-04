"""
LinkedIn carousel — Deep Research Agent
Square slides (612x612 pt = 8.5x8.5 in), clean research format.
"""
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import os

OUT = os.path.join(os.path.dirname(__file__), "deep_research_agent_carousel.pdf")

# ── Canvas size (square) ─────────────────────────────────────────────────────
W = H = 612.0   # 8.5 × 8.5 inches

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY    = HexColor("#0D1B2A")
BLUE    = HexColor("#1A56A0")
LBLUE   = HexColor("#EBF4FB")
MID     = HexColor("#3A3A4A")
MUTED   = HexColor("#888899")
WHITE   = white
DIVIDER = HexColor("#D0DCF0")
GREEN   = HexColor("#1A7A4A")
AMBER   = HexColor("#B45309")

# ── Helpers ──────────────────────────────────────────────────────────────────
def new_page(c):
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)

def accent_bar(c, color=BLUE, x=48, y=None, w=None, h=4):
    y = y or (H - 60)
    w = w or (W - 96)
    c.setFillColor(color)
    c.rect(x, y, w, h, fill=1, stroke=0)

def slide_number(c, n, total=9):
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawRightString(W - 48, 24, f"{n} / {total}")

def footer_line(c, text="Darshil Shukla  |  Deep Research Agent  |  github.com/DarshilShukla26/deep-research-agent"):
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED)
    c.drawCentredString(W / 2, 16, text)

def heading(c, text, y, size=26, color=NAVY, font="Helvetica-Bold"):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(48, y, text)

def subheading(c, text, y, size=13, color=BLUE):
    c.setFont("Helvetica-Bold", size)
    c.setFillColor(color)
    c.drawString(48, y, text)

def body(c, text, y, size=12, color=MID, x=48):
    c.setFont("Helvetica", size)
    c.setFillColor(color)
    c.drawString(x, y, text)

def bullet_row(c, text, y, size=12, color=MID, indent=64):
    c.setFont("Helvetica", 8)
    c.setFillColor(BLUE)
    c.drawString(48, y + 1, "\u2022")
    c.setFont("Helvetica", size)
    c.setFillColor(color)
    c.drawString(indent, y, text)

def tag_box(c, text, x, y, bg=LBLUE, fg=BLUE):
    c.setFont("Helvetica-Bold", 9)
    tw = c.stringWidth(text, "Helvetica-Bold", 9)
    pad = 8
    c.setFillColor(bg)
    c.roundRect(x, y - 2, tw + pad * 2, 18, 4, fill=1, stroke=0)
    c.setFillColor(fg)
    c.drawString(x + pad, y + 2, text)

def stat_block(c, value, label, x, y, vsize=36, lsize=10):
    c.setFont("Helvetica-Bold", vsize)
    c.setFillColor(BLUE)
    c.drawCentredString(x, y, value)
    c.setFont("Helvetica", lsize)
    c.setFillColor(MUTED)
    c.drawCentredString(x, y - 16, label)

def h_rule(c, y, x=48, color=DIVIDER):
    c.setStrokeColor(color)
    c.setLineWidth(0.5)
    c.line(x, y, W - x, y)

def pill(c, text, x, y, bg=NAVY, fg=WHITE):
    c.setFont("Helvetica-Bold", 9)
    tw = c.stringWidth(text, "Helvetica-Bold", 9)
    pad = 10
    c.setFillColor(bg)
    c.roundRect(x, y - 3, tw + pad * 2, 20, 10, fill=1, stroke=0)
    c.setFillColor(fg)
    c.drawString(x + pad, y + 1, text)

# ── SLIDES ───────────────────────────────────────────────────────────────────
c = canvas.Canvas(OUT, pagesize=(W, H))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Cover
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
c.setFillColor(NAVY)
c.rect(0, H - 220, W, 220, fill=1, stroke=0)

c.setFont("Helvetica-Bold", 9)
c.setFillColor(HexColor("#7AADDD"))
c.drawString(48, H - 52, "RESEARCH MEMO")

c.setFont("Helvetica-Bold", 32)
c.setFillColor(WHITE)
c.drawString(48, H - 100, "Building a Research Agent")
c.drawString(48, H - 136, "That Remembers")

c.setFont("Helvetica", 13)
c.setFillColor(HexColor("#A8C4E0"))
c.drawString(48, H - 168, "3-Layer Memory + Token Budget Guard + n8n Routing")

c.setFont("Helvetica", 10)
c.setFillColor(HexColor("#7AADDD"))
c.drawString(48, H - 200, "What I built, what I measured, and what I learned")

c.setFillColor(WHITE)
c.rect(0, H - 224, W, 4, fill=1, stroke=0)

y = H - 280
for tag, tx in [("Python", 48), ("ChromaDB", 118), ("Anthropic SDK", 198), ("FastAPI", 308), ("n8n", 378)]:
    pill(c, tag, tx, y, bg=BLUE, fg=WHITE)

y = 180
c.setFont("Helvetica", 10)
c.setFillColor(MUTED)
c.drawString(48, y, "Darshil Shukla")
h_rule(c, y - 12)
footer_line(c)
slide_number(c, 1)
c.showPage()

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — The Problem
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
accent_bar(c, y=H - 52)
heading(c, "The Problem", H - 86)
c.setFont("Helvetica", 12)
c.setFillColor(MID)
c.drawString(48, H - 114, "Most LLM usage in research workflows is broken in 3 ways:")

problems = [
    ("No memory",        "Every query starts from zero — no context, no history"),
    ("No cost control",  "One complex prompt can cost $2–5 with no guardrails"),
    ("No audit trail",   "You can't prove what was asked, what it cost, or how good the answer was"),
]
y = H - 168
for title, desc in problems:
    c.setFillColor(LBLUE)
    c.roundRect(48, y - 8, W - 96, 52, 6, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(NAVY)
    c.drawString(64, y + 26, title)
    c.setFont("Helvetica", 11)
    c.setFillColor(MID)
    c.drawString(64, y + 8, desc)
    y -= 72

h_rule(c, 44)
footer_line(c)
slide_number(c, 2)
c.showPage()

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — The Architecture
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
accent_bar(c)
heading(c, "The Architecture", H - 86)
c.setFont("Helvetica", 11)
c.setFillColor(MUTED)
c.drawString(48, H - 110, "A 5-layer pipeline where every layer has one job")

layers = [
    (BLUE,              "Router + Budget Guard",  "Hard token cap enforced before every API call"),
    (HexColor("#2E86AB"), "Memory Layer",          "Vector RAG  +  Episodic Buffer  +  Summary Cascade"),
    (HexColor("#1A7A4A"), "Context Assembler",     "Greedy bin-packing — fits memory into remaining budget"),
    (HexColor("#6B4C9A"), "Claude (tool use)",     "Haiku for search/decompose  |  Opus for final synthesis"),
    (HexColor("#B45309"), "Evaluation Logger",     "Tokens, cost, strategies, self-score  →  evaluation.md"),
]
y = H - 156
arrow_x = W / 2
for i, (color, title, desc) in enumerate(layers):
    c.setFillColor(color)
    c.roundRect(48, y - 6, W - 96, 44, 5, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(WHITE)
    c.drawString(64, y + 22, title)
    c.setFont("Helvetica", 9.5)
    c.setFillColor(HexColor("#DDEEFF"))
    c.drawString(64, y + 7, desc)
    y -= 52
    if i < len(layers) - 1:
        c.setFillColor(DIVIDER)
        c.rect(arrow_x - 1, y + 2, 2, 6, fill=1, stroke=0)

h_rule(c, 44)
footer_line(c)
slide_number(c, 3)
c.showPage()

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — 3-Layer Memory
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
accent_bar(c, color=HexColor("#2E86AB"))
heading(c, "The 3-Layer Memory System", H - 86, size=22)

layers = [
    ("Vector RAG",       "Long-term",   "ChromaDB semantic search — persists across sessions.\nEvery answer gets stored back as future knowledge."),
    ("Episodic Buffer",  "Short-term",  "Last 10 turns of the current session in RAM.\nPrevents re-asking the same sub-questions."),
    ("Summary Cascade",  "Compressed",  "When buffer fills, oldest 5 turns are compressed\ninto a rolling 200-word summary by Claude."),
]
y = H - 152
for title, badge, desc in layers:
    c.setFillColor(LBLUE)
    c.roundRect(48, y - 14, W - 96, 72, 6, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(NAVY)
    c.drawString(64, y + 40, title)
    tag_box(c, badge, 220, y + 36)
    lines = desc.split("\n")
    c.setFont("Helvetica", 10.5)
    c.setFillColor(MID)
    c.drawString(64, y + 20, lines[0])
    c.drawString(64, y + 5, lines[1])
    y -= 88

h_rule(c, 44)
footer_line(c)
slide_number(c, 4)
c.showPage()

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Key Trade-offs
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
accent_bar(c, color=HexColor("#6B4C9A"))
heading(c, "Key Design Trade-offs", H - 86, size=24)
c.setFont("Helvetica", 11)
c.setFillColor(MUTED)
c.drawString(48, H - 112, "Every architecture choice had an explicit reason and an accepted cost")

tradeoffs = [
    ("ChromaDB over Pinecone",     "Zero infra, <5ms latency, free",           "Weaker embedding model"),
    ("Ring buffer over full history", "O(1) context cost per call",             "Old turns lost if cascade skipped"),
    ("Greedy packing over DP",     "O(n) speed in the hot path",                "~2% sub-optimal packing"),
    ("Haiku for tool calls",       "60-70% cost reduction",                     "Slightly less reasoning depth"),
    ("tiktoken for estimation",    "Pre-call budget check",                     "+-2-5% vs Claude tokeniser"),
]
y = H - 156
c.setFont("Helvetica-Bold", 9)
c.setFillColor(BLUE)
c.drawString(48, y, "DECISION")
c.drawString(252, y, "WHY")
c.drawString(432, y, "TRADE-OFF")
y -= 14
h_rule(c, y + 4)
y -= 10

for decision, why, cost in tradeoffs:
    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(NAVY)
    c.drawString(48, y, decision)
    c.setFont("Helvetica", 9.5)
    c.setFillColor(GREEN)
    c.drawString(252, y, why)
    c.setFillColor(AMBER)
    c.drawString(432, y, cost)
    y -= 6
    h_rule(c, y, color=HexColor("#EEEEEE"))
    y -= 18

h_rule(c, 44)
footer_line(c)
slide_number(c, 5)
c.showPage()

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Real Numbers
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
accent_bar(c, color=GREEN)
heading(c, "Findings — Real Run Data", H - 86, size=24)
c.setFont("Helvetica", 11)
c.setFillColor(MUTED)
c.drawString(48, H - 112, "Every run logged to evaluation.md — these are actual numbers")

# Stats row
y = H - 164
for val, lbl, x in [
    ("$0.07",  "cheapest\nsuccessful run",   120),
    ("19.4%",  "lowest budget\nutilisation", 306),
    ("5",      "sub-questions on\nfirst RLHF query", 490),
]:
    lines = lbl.split("\n")
    c.setFont("Helvetica-Bold", 30)
    c.setFillColor(BLUE)
    c.drawCentredString(x, y, val)
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawCentredString(x, y - 16, lines[0])
    c.drawCentredString(x, y - 27, lines[1])

y = H - 236
h_rule(c, y)
y -= 20

headers = ["Query", "Tokens", "Cost", "Util%"]
col_x   = [48, 260, 370, 470]
c.setFont("Helvetica-Bold", 9)
c.setFillColor(BLUE)
for hdr, cx in zip(headers, col_x):
    c.drawString(cx, y, hdr)
y -= 14
h_rule(c, y + 4)
y -= 10

rows = [
    ("RLHF vs DPO",          "9,713",  "$0.11",  "19.4%"),
    ("Transformer architecture","13,370","$0.11", "89.1%"),
    ("Evolution of AI",      "13,620",  "$0.11",  "27.2%"),
    ("UCL season (web search)","11,046","$0.07",  "22.1%"),
]
for i, (q, tok, cost, util) in enumerate(rows):
    if i % 2 == 0:
        c.setFillColor(HexColor("#F7FAFD"))
        c.rect(46, y - 5, W - 92, 18, fill=1, stroke=0)
    c.setFont("Helvetica", 9.5)
    c.setFillColor(NAVY)
    c.drawString(col_x[0], y, q)
    c.setFillColor(MID)
    c.drawString(col_x[1], y, tok)
    c.drawString(col_x[2], y, cost)
    c.setFillColor(GREEN if float(util.replace("%","")) < 50 else AMBER)
    c.drawString(col_x[3], y, util)
    y -= 22

h_rule(c, 44)
footer_line(c)
slide_number(c, 6)
c.showPage()

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — Model Split Strategy
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
accent_bar(c, color=HexColor("#B45309"))
heading(c, "The Model Split Strategy", H - 86, size=24)
c.setFont("Helvetica", 11)
c.setFillColor(MUTED)
c.drawString(48, H - 112, "Using two Claude models for different jobs cut cost by ~65%")

# Two columns
col_w = (W - 112) / 2
lx, rx = 48, 48 + col_w + 16

def model_box(c, x, y, w, title, badge, badge_bg, items, note):
    c.setFillColor(LBLUE)
    c.roundRect(x, y - 10, w, 210, 8, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(NAVY)
    c.drawString(x + 16, y + 182, title)
    tag_box(c, badge, x + 16, y + 158, bg=badge_bg, fg=WHITE)
    by = y + 130
    for item in items:
        bullet_row(c, item, by, size=10, indent=x + 36)
        by -= 22
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawString(x + 16, y + 4, note)

model_box(c, lx, H - 352, col_w,
    "claude-haiku-4-5", "Fast + Cheap", HexColor("#2E86AB"),
    ["Decompose questions", "Search knowledge base", "Search the web", "Self-score answers"],
    "$1 input / $5 output per 1M tokens"
)
model_box(c, rx, H - 352, col_w,
    "claude-opus-4-6", "High Quality", HexColor("#6B4C9A"),
    ["Final answer synthesis", "Answer formatting + polish", "Summary cascade compression"],
    "$5 input / $25 output per 1M tokens"
)

# Saving callout
c.setFillColor(HexColor("#FFF8E7"))
c.roundRect(48, H - 400, W - 96, 40, 6, fill=1, stroke=0)
c.setFont("Helvetica-Bold", 11)
c.setFillColor(AMBER)
c.drawString(64, H - 376, "Result:")
c.setFont("Helvetica", 11)
c.setFillColor(MID)
c.drawString(116, H - 376, "~65% cost reduction vs using Opus for everything, no quality drop on tool-use steps")

h_rule(c, 44)
footer_line(c)
slide_number(c, 7)
c.showPage()

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Key Learnings
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
accent_bar(c)
heading(c, "Key Learnings", H - 86)
c.setFont("Helvetica", 11)
c.setFillColor(MUTED)
c.drawString(48, H - 112, "What surprised me building this")

learnings = [
    ("Token counting is not the same as token charging",
     "tiktoken estimates diverge 2-5% from Claude's actual count. You need a safety margin."),
    ("Memory strategy matters more than model quality",
     "A well-assembled 8k context with the right chunks beats a 50k dump of everything."),
    ("Self-scoring is surprisingly honest",
     "Haiku consistently scored knowledge-gap answers lower (3/5) and substantive answers higher (4-5/5)."),
    ("n8n routing pays off fast",
     "Simple queries auto-routed to 15k budget saved ~$0.03 each. At scale that compounds quickly."),
    ("ChromaDB is underrated for local RAG",
     "Sub-5ms retrieval, zero infra, persistent across restarts. Only limitation is embedding quality."),
]

y = H - 152
for i, (title, desc) in enumerate(learnings):
    c.setFillColor(LBLUE if i % 2 == 0 else WHITE)
    c.roundRect(48, y - 8, W - 96, 50, 5, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 10.5)
    c.setFillColor(NAVY)
    c.drawString(64, y + 26, title)
    c.setFont("Helvetica", 9.5)
    c.setFillColor(MID)
    c.drawString(64, y + 10, desc)
    y -= 62

h_rule(c, 44)
footer_line(c)
slide_number(c, 8)
c.showPage()

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — CTA / Closing
# ─────────────────────────────────────────────────────────────────────────────
new_page(c)
c.setFillColor(NAVY)
c.rect(0, 0, W, H, fill=1, stroke=0)

c.setFont("Helvetica-Bold", 10)
c.setFillColor(HexColor("#7AADDD"))
c.drawString(48, H - 60, "OPEN SOURCE")

c.setFont("Helvetica-Bold", 28)
c.setFillColor(WHITE)
c.drawString(48, H - 108, "Everything is on GitHub.")
c.drawString(48, H - 144, "Clone it, break it,")
c.drawString(48, H - 180, "improve it.")

c.setFont("Helvetica", 13)
c.setFillColor(HexColor("#A8C4E0"))
c.drawString(48, H - 220, "github.com/DarshilShukla26/deep-research-agent")

c.setFillColor(HexColor("#1A56A0"))
c.rect(0, H - 228, W, 4, fill=1, stroke=0)

stack_items = ["Python", "ChromaDB", "Anthropic SDK", "FastAPI", "n8n", "DuckDuckGo Search"]
tx = 48
ty = H - 300
for item in stack_items:
    pill(c, item, tx, ty, bg=HexColor("#1A3A5C"), fg=HexColor("#A8C4E0"))
    c.setFont("Helvetica-Bold", 9)
    tw = c.stringWidth(item, "Helvetica-Bold", 9)
    tx += tw + 38
    if tx > W - 150:
        tx = 48
        ty -= 34

c.setFont("Helvetica", 10)
c.setFillColor(HexColor("#5A8ABB"))
c.drawString(48, 200, "If you're working on memory systems, cost-aware agents,")
c.drawString(48, 184, "or n8n automation — I'd love to connect.")

c.setFont("Helvetica-Bold", 11)
c.setFillColor(WHITE)
c.drawString(48, 148, "Darshil Shukla")
c.setFont("Helvetica", 10)
c.setFillColor(HexColor("#7AADDD"))
c.drawString(48, 132, "Graduate AI Engineer")

slide_number(c, 9)
c.save()

print(f"Done: {OUT}")
