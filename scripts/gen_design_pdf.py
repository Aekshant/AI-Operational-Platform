"""Generates docs/ForgeAI-Design.pdf (HLD, LLD, flows). Needs: uv pip install reportlab. Run: python scripts/gen_design_pdf.py"""

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = str(Path(__file__).resolve().parents[1] / "docs/ForgeAI-Design.pdf")
F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Sans", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", F + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")

INK = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")
LINE = colors.HexColor("#d1d5db")
ACCENT = colors.HexColor("#2f4fbf")
C_CP = colors.HexColor("#e8edfc")    # control plane
C_DP = colors.HexColor("#e6f6ee")    # data plane
C_OBS = colors.HexColor("#fdf1e3")   # observability
C_AG = colors.HexColor("#f8e9f3")    # agent
C_ST = colors.HexColor("#f1f2f4")    # storage / external
C_BAD = colors.HexColor("#fde8e8")
B_CP, B_DP, B_OBS, B_AG = (colors.HexColor(c) for c in ("#5b74d6", "#3f9d6f", "#d08a32", "#b3588f"))

W = A4[0] - 36 * mm  # content width

ss = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=26, leading=32, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=12.5, leading=18, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=17, leading=22, textColor=INK, spaceBefore=4,
                         spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=12,
                         spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=10.5, leading=14, textColor=ACCENT, spaceBefore=8,
                         spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.4, leading=13.6, textColor=INK, spaceAfter=5,
                           alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=8.2, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.2, leading=10.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8.2, leading=10.8, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.8, leading=10.4, textColor=INK,
                           backColor=colors.HexColor("#f6f7f9"), borderPadding=(5, 6, 5, 6), spaceBefore=3,
                           spaceAfter=8),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=9.4, leading=13.4, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2),
    "caption": ParagraphStyle("caption", fontName="Sans", fontSize=8, leading=10, textColor=MUTED, spaceBefore=2,
                              spaceAfter=10),
}


def P(t, s="body"):
    return Paragraph(t, ss[s])


def bullets(items):
    return [Paragraph(i, ss["bullet"], bulletText="•") for i in items]


def code(t):
    return Paragraph(t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                     .replace("  ", "&nbsp;&nbsp;"), ss["code"])


def table(rows, widths, header=True, mono_cols=()):
    data = []
    for r_i, r in enumerate(rows):
        row = []
        for c_i, c in enumerate(r):
            if isinstance(c, str):
                style = "cellb" if header and r_i == 0 else "cell"
                txt = c
                if c_i in mono_cols and r_i > 0:
                    txt = f'<font name="Mono" size="7.6">{c}</font>'
                row.append(Paragraph(txt, ss[style]))
            else:
                row.append(c)
        data.append(row)
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0f4")), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
    t.setStyle(TableStyle(st))
    return t


def caption(t):
    return P(t, "caption")


# ----------------------------------------------------------------------------- drawing primitives
def box(d, x, y, w, h, title, sub=None, fill=C_ST, stroke=MUTED, fs=8.2, bold=True, sub_fs=6.8):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.8))
    lines = sub.split("\n") if sub else []
    total = fs + len(lines) * (sub_fs + 2.2)
    ty = y + h / 2 + total / 2 - fs + 1
    d.add(String(x + w / 2, ty, title, fontName="Sans-Bold" if bold else "Sans", fontSize=fs, fillColor=INK,
                 textAnchor="middle"))
    for i, ln in enumerate(lines):
        d.add(String(x + w / 2, ty - (i + 1) * (sub_fs + 2.2), ln, fontName="Sans", fontSize=sub_fs,
                     fillColor=MUTED, textAnchor="middle"))


def region(d, x, y, w, h, label, fill, stroke, right=False):
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=stroke, strokeWidth=0.8,
               strokeDashArray=[3, 2]))
    if right:
        d.add(String(x + w - 7, y + h - 11, label, fontName="Sans-Bold", fontSize=7.6, fillColor=stroke,
                     textAnchor="end"))
    else:
        d.add(String(x + 7, y + h - 11, label, fontName="Sans-Bold", fontSize=7.6, fillColor=stroke))


def arrow(d, x1, y1, x2, y2, label=None, dashed=False, color=INK, lx=0, ly=3, fs=6.6, head=True, anchor="middle"):
    import math

    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.8, strokeDashArray=[3, 2] if dashed else None))
    if head:
        a = math.atan2(y2 - y1, x2 - x1)
        s = 5
        p1 = (x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
        p2 = (x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4))
        d.add(Polygon([x2, y2, *p1, *p2], fillColor=color, strokeColor=color, strokeWidth=0.4))
    if label:
        globals()["label"](d, (x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fs=fs, color=MUTED, anchor=anchor)


def poly_arrow(d, pts, label=None, dashed=False, color=INK, lpos=None, fs=6.6):
    for (a, b), (c, e) in zip(pts[:-2], pts[1:-1], strict=True):
        d.add(Line(a, b, c, e, strokeColor=color, strokeWidth=0.8, strokeDashArray=[3, 2] if dashed else None))
    arrow(d, *pts[-2], *pts[-1], dashed=dashed, color=color)
    if label and lpos:
        globals()["label"](d, lpos[0], lpos[1], label, fs=fs, color=MUTED)


def label(d, x, y, text, fs=6.6, color=INK, anchor="middle"):
    w = pdfmetrics.stringWidth(text, "Sans", fs)
    lx = x - w / 2 if anchor == "middle" else x
    d.add(Rect(lx - 2, y - 2, w + 4, fs + 2.5, fillColor=colors.white, strokeColor=None))
    d.add(String(x, y, text, fontName="Sans", fontSize=fs, fillColor=color, textAnchor=anchor))


def fig(heading, drawing, cap, intro=None):
    parts = ([P(heading, "h2")] if heading else []) + ([P(intro)] if intro else []) + [drawing, caption(cap)]
    return KeepTogether(parts)


# ----------------------------------------------------------------------------- sequence diagrams
def sequence(parts, msgs, width=None, title=None, step=19, colors_map=None):
    """parts: [(key, label)]; msgs: (frm, to, label, kind) kind in solid|reply|self|note|div."""
    width = width or W
    n = len(parts)
    top_pad, head_h = 8, 26
    h = top_pad + head_h + 14 + step * len(msgs) + 16
    d = Drawing(width, h)
    xs = {}
    col = width / n
    for i, (k, lab) in enumerate(parts):
        cx = col * i + col / 2
        xs[k] = cx
        bw = min(col - 8, 104)
        fill = (colors_map or {}).get(k, C_ST)
        lines = lab.split("\n")
        box(d, cx - bw / 2, h - top_pad - head_h, bw, head_h, lines[0], "\n".join(lines[1:]) or None, fill=fill,
            fs=7.4)
        d.add(Line(cx, h - top_pad - head_h, cx, 8, strokeColor=LINE, strokeWidth=0.8, strokeDashArray=[2, 2]))
    y = h - top_pad - head_h - 14
    for m in msgs:
        frm, to, lab_text, kind = (m + ("solid",))[:4] if len(m) == 3 else m
        label_ = lab_text  # noqa
        if kind == "div":
            d.add(Rect(4, y - 6, width - 8, 12, fillColor=colors.HexColor("#f6f7f9"), strokeColor=None))
            d.add(String(width / 2, y - 2.5, lab_text, fontName="Sans-Bold", fontSize=6.8, fillColor=ACCENT,
                         textAnchor="middle"))
        elif kind == "note":
            x = xs[frm]
            tw = pdfmetrics.stringWidth(lab_text, "Sans", 6.6) + 10
            nx = min(max(x - tw / 2, 2), width - tw - 2)
            d.add(Rect(nx, y - 6, tw, 12, fillColor=colors.HexColor("#fffbe6"),
                       strokeColor=colors.HexColor("#e5c96a"), strokeWidth=0.5))
            d.add(String(nx + tw / 2, y - 2.5, lab_text, fontName="Sans", fontSize=6.6, fillColor=INK,
                         textAnchor="middle"))
        elif kind == "self" or frm == to:
            x = xs[frm]
            d.add(Line(x, y + 2, x + 16, y + 2, strokeColor=INK, strokeWidth=0.7))
            d.add(Line(x + 16, y + 2, x + 16, y - 6, strokeColor=INK, strokeWidth=0.7))
            arrow(d, x + 16, y - 6, x + 1, y - 6)
            label(d, x + 20, y - 3, lab_text, anchor="start")
        else:
            x1, x2 = xs[frm], xs[to]
            dashed = kind == "reply"
            color = colors.HexColor("#b42318") if kind == "bad" else (MUTED if dashed else INK)
            arrow(d, x1, y - 4, x2 + (-1 if x2 > x1 else 1), y - 4, dashed=dashed, color=color)
            label(d, (x1 + x2) / 2, y - 1, lab_text, color=color if kind == "bad" else INK)
        y -= step
    return d


# ----------------------------------------------------------------------------- page decoration
def on_page(c, doc):
    c.saveState()
    c.setFont("Sans", 7.5)
    c.setFillColor(MUTED)
    if doc.page > 1:
        c.drawString(18 * mm, 12 * mm, "ForgeAI: System Design (HLD · LLD · Flows)")
        c.drawRightString(A4[0] - 18 * mm, 12 * mm, f"{doc.page}")
        c.setStrokeColor(LINE)
        c.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
    c.restoreState()


story = []
S = story.append

# ============================================================================= COVER
S(Spacer(1, 50 * mm))
S(P("ForgeAI", "title"))
S(Spacer(1, 4))
S(P("System design: high-level design, low-level design and flows", "subtitle"))
S(Spacer(1, 14))
S(P("A self-hosted platform for fine-tuning, evaluating, deploying, serving and operating small language models, "
    "operated by an AI SRE agent that diagnoses incidents and remediates them after human approval.", "body"))
S(Spacer(1, 22))
S(table([
    ["Field", "Value"],
    ["Document", "System design: HLD, LLD, flows"],
    ["Version", "0.1"],
    ["Date", "5 October 2026"],
    ["Status", "Implemented. Local stack verified end to end; GPU / Kubernetes paths not yet exercised"],
    ["Codebase", "forge/ (Python 3.11): ~4,100 lines across 24 modules; 35 tests"],
    ["Audience", "Engineers reviewing or extending the platform"],
], [0.25, 0.75]))
S(Spacer(1, 20))
S(P("<b>Contents</b>", "h3"))
for line in ["1. Overview", "2. High-level design (HLD)", "3. Low-level design (LLD)", "4. Flows",
             "5. Verification status and known limits"]:
    S(P(line, "body"))
S(PageBreak())

# ============================================================================= 1. OVERVIEW
S(P("1. Overview", "h1"))
S(P("ForgeAI turns the model lifecycle into one system: <b>train → evaluate → register → deploy → serve → observe → "
    "diagnose → approve → remediate → verify</b>. The agent is not the product. It is the platform's operator, "
    "and it acts only through the same API, RBAC, audit log and idempotency guarantees as a human operator."))
S(P("1.1 Goals", "h2"))
story.extend(bullets([
    "Produce a better model from a dataset (LoRA / QLoRA SFT) with full data lineage.",
    "Answer <i>“is v2 actually better than v1?”</i> across quality, latency and cost, and turn the answer into an "
    "explicit ship decision.",
    "Serve models through a reliability-aware, OpenAI-compatible gateway: canary routing, rate limits, admission "
    "control, circuit breaking, retry and fallback.",
    "Make every layer observable: SLIs (TTFT, TPOT, tokens, cost), model-server and GPU metrics, traces and logs.",
    "Detect SLO violations, diagnose them with evidence, and remediate safely with a human in the loop.",
]))
S(P("1.2 Non-goals (current version)", "h2"))
story.extend(bullets([
    "Multi-region or multi-cluster serving; pretraining; RLHF / preference optimisation.",
    "Fully autonomous remediation without approval.",
    "Pod lifecycle management. The deploy controller owns <i>routing</i>; Kubernetes (Helm/KEDA) owns pods.",
]))
S(P("1.3 Terminology", "h2"))
S(table([
    ["Term", "Meaning"],
    ["Deployment", "A named serving endpoint (e.g. <i>qwen-farm</i>) with a stable version and an optional canary version"],
    ["Model version", "Immutable registry entry: base model + adapter + dataset hash + eval metrics + lifecycle stage"],
    ["TTFT / TPOT", "Time to first token / time per output token: the two latencies users perceive in streaming"],
    ["Revision", "Monotonic counter on a deployment, bumped by every mutation; used for optimistic concurrency"],
    ["Incident", "Human-facing record of one run of the agent's state machine, keyed by incident id"],
    ["Ship gate", "Policy comparing two evaluated versions → SHIP_CANARY / NEEDS_REVIEW / BLOCK"],
], [0.2, 0.8]))
S(PageBreak())

# ============================================================================= 2. HLD
S(P("2. High-level design", "h1"))
d = Drawing(W, 230)
cx, cy, cw, ch = W / 2 - 95, 80, 190, 70
box(d, cx, cy, cw, ch, "ForgeAI platform", "control plane · gateway · model servers\nobservability · AI SRE",
    fill=C_CP, stroke=B_CP, fs=10)
actors = [
    (6, 175, "ML engineer", "train · evaluate · ship", (cx, cy + ch - 10), "forge CLI / API"),
    (6, 20, "Tenant applications", "OpenAI-compatible clients", (cx, cy + 12), "/v1/chat/completions"),
    (W - 136, 175, "On-call engineer", "approve / reject actions", (cx + cw, cy + ch - 10), "UI · CLI decisions"),
    (W - 136, 20, "External agents", "Claude Code / Desktop (MCP)", (cx + cw, cy + 12), "MCP tools"),
]
for x, y, t, s, (tx, ty), lab in actors:
    box(d, x, y, 130, 36, t, s, fill=C_ST)
    sx = x + 130 if x < W / 2 else x
    arrow(d, sx, y + 18, tx, ty, lab, ly=4)
box(d, W / 2 - 75, 190, 150, 32, "Hugging Face Hub", "base models (download)", fill=C_ST)
arrow(d, W / 2, 190, W / 2, cy + ch, "weights", lx=22)
box(d, W / 2 - 75, 4, 150, 32, "Anthropic API (optional)", "Claude for investigations", fill=C_ST)
arrow(d, W / 2, cy, W / 2, 36, "tool-use loop", lx=28)
S(fig("2.1 System context", d, "Figure 1. System context. Everything inside the platform box is self-hosted; Claude "
      "is optional and the agent falls back to a deterministic runbook without it."))
S(PageBreak())

d = Drawing(W, 380)
# client layer
box(d, 4, 344, 150, 30, "Web UI", "tiles · incidents · approvals", fill=C_ST)
box(d, W / 2 - 75, 344, 150, 30, "forge CLI", "train · eval · deploy · agent", fill=C_ST)
box(d, W - 154, 344, 150, 30, "MCP server", "read tools + request_action", fill=C_ST)
# control plane
region(d, 2, 222, W - 4, 112, "CONTROL PLANE  (forge serve controlplane · FastAPI · SQLite WAL)", C_CP, B_CP)
cpw = (W - 30) / 5
labels = [("Training", "dataset · LoRA/QLoRA\nmemory planner"), ("Evaluation", "quality · latency · cost\nship gate"),
          ("Model registry", "versions · lineage\nlifecycle stages"),
          ("Deploy controller", "canary · promote\nrollback · revisions"),
          ("Incidents + AI SRE", "LangGraph state\nmachine · HITL")]
for i, (t, s) in enumerate(labels):
    box(d, 8 + i * (cpw + 3.5), 232, cpw, 74, t, s, fill=C_AG if i == 4 else colors.white,
        stroke=B_AG if i == 4 else B_CP)
for i in range(3):
    arrow(d, 8 + (i + 1) * (cpw + 3.5) - 3.5, 269, 8 + (i + 1) * (cpw + 3.5), 269)
for x in (4 + 75, W / 2, W - 79):
    arrow(d, x, 344, x, 334)
# data plane
region(d, 2, 104, W - 4, 108, "DATA PLANE", C_DP, B_DP)
box(d, 10, 116, W * 0.55, 80, "Inference gateway",
    "auth/RBAC → rate limit → guardrails → weighted routing\n→ circuit breaker → bulkhead + priority queue\n"
    "→ TTFT timeout · retry elsewhere · fallback → streaming relay", fill=colors.white, stroke=B_DP)
mx = W * 0.55 + 22
mw = (W - mx - 22) / 2 - 3
box(d, mx, 156, mw, 40, "Model server v2", "stable · vLLM + LoRA", fill=colors.white, stroke=B_DP)
box(d, mx + mw + 6, 156, mw, 40, "Model server v3", "canary · vLLM + LoRA", fill=colors.white, stroke=B_DP)
box(d, mx, 116, 2 * mw + 6, 30, "simulator (no GPU)", "same OpenAI API + vllm:* / DCGM metrics", fill=C_ST,
    fs=7.4)
arrow(d, 10 + W * 0.55, 176, mx, 176)
arrow(d, W * 0.33, 222, W * 0.33, 196, "GET /api/routing (poll 2s)", lx=52, ly=-2, dashed=True)
# observability
region(d, 2, 4, W - 4, 80, "OBSERVABILITY", C_OBS, B_OBS, right=True)
ow = (W - 30) / 6
for i, (t, s) in enumerate([("Prometheus", "scrape 5s · rules"), ("Alertmanager", "route criticals"),
                            ("OTel Collector", "OTLP traces+logs"), ("Tempo", "traces"), ("Loki", "logs"),
                            ("Grafana", "dashboard as code")]):
    box(d, 8 + i * (ow + 2.8), 12, ow, 46, t, s, fill=colors.white, stroke=B_OBS, fs=7.6)
arrow(d, 8 + ow, 35, 8 + ow + 2.8, 35)
arrow(d, 8 + ow / 2, 116, 8 + ow / 2, 58, dashed=True)
label(d, 8 + ow / 2 + 4, 91, "scrape /metrics", color=MUTED, anchor="start")
am_x = 8 + (ow + 2.8) + ow / 2
poly_arrow(d, [(am_x, 58), (am_x, 94), (W - 10, 94), (W - 10, 232)], dashed=True, color=B_AG)
label(d, (am_x + W) / 2 + 20, 91, "alert webhook → incident (AI SRE)", color=B_AG)
S(fig("2.2 Architecture", d, "Figure 2. Logical architecture. Solid arrows are request paths; dashed arrows are "
      "asynchronous (polling, scraping, alerting)."))
S(PageBreak())

S(P("2.3 Components", "h2"))
S(table([
    ["Component", "Responsibility", "Technology", "Scaling / state"],
    ["Training", "Dataset validation and lineage; SFT with LoRA/QLoRA; GPU memory plan", "transformers, peft, "
     "bitsandbytes", "Batch job on a GPU node; writes immutable run dirs"],
    ["Evaluation", "Quality, TTFT, p95, tokens/s, $/request on an eval set; ship gate", "httpx (streaming)",
     "Stateless; reports to registry"],
    ["Model registry", "Immutable versions, metrics, stages, transition history", "SQLite (WAL); MLflow mirror",
     "Single writer"],
    ["Deploy controller", "Desired routing state; canary/promote/rollback; revisions; idempotency; events",
     "SQLite", "Single writer"],
    ["Control-plane API", "REST for all of the above; incidents; alert webhook; UI", "FastAPI", "1 replica (SQLite)"],
    ["Inference gateway", "OpenAI-compatible serving with reliability controls and SLI metrics", "FastAPI, httpx",
     "Stateless; HPA on CPU"],
    ["Model servers", "Token generation with continuous batching and LoRA adapters", "vLLM (simulator locally)",
     "One Deployment per version; KEDA on queue"],
    ["AI SRE", "Investigate with tools, propose action, wait for approval, execute, verify",
     "LangGraph + SQLite checkpoints; Claude or runbook", "Thread pool in control plane"],
    ["MCP server", "Expose read tools and request_action to external agents", "mcp 2.x (MCPServer)",
     "stdio / streamable-http"],
    ["Observability", "Metrics, alerts, traces, logs, dashboards",
     "Prometheus, Alertmanager, OTel, Tempo, Loki, Grafana", "Standard deployments"],
], [0.16, 0.38, 0.22, 0.24]))

S(P("2.4 Control plane / data plane split", "h2"))
S(P("The control plane owns <b>desired state</b> (which versions serve which share of traffic). The gateway "
    "converges on it by polling <font name='Mono'>/api/routing</font> every 2 s, so rollouts are "
    "<b>eventually consistent</b> with a bounded lag. If the control plane is unreachable the gateway keeps serving "
    "its <b>last-known-good</b> routing table and exports "
    "<font name='Mono'>forge_routing_stale_seconds</font>; an alert fires after 30 s of staleness. The data plane "
    "never depends on the control plane being up to serve traffic."))

S(P("2.5 Deployment views", "h2"))
S(table([
    ["Concern", "Local (docker-compose)", "Kubernetes (Helm chart)"],
    ["Model servers", "3 simulator containers (v1, v2, v3) with per-version capacity profiles",
     "vLLM Deployment per version: --enable-lora --lora-modules, GPU limits, nodeSelector, tolerations"],
    ["Version labels on metrics", "Static scrape labels in prometheus.yml", "PodMonitor relabels pod labels "
     "forgeai.dev/model, forgeai.dev/version"],
    ["Autoscaling", "n/a", "Gateway HPA (CPU); KEDA ScaledObject on vllm:num_requests_waiting per version"],
    ["Availability", "restart: unless-stopped", "Gateway PDB, topology spread, preStop drain, 60 s grace"],
    ["Isolation", "Compose network", "NetworkPolicy: model servers reachable only from gateway, control plane, "
     "monitoring"],
    ["Alerts", "deploy/prometheus/alerts.yml", "PrometheusRule from the same file (test keeps copies identical)"],
    ["Secrets", ".env (dev keys)", "Secret / existingSecret for API keys; optional Anthropic + HF token secrets"],
], [0.18, 0.37, 0.45]))

S(P("2.6 Non-functional requirements", "h2"))
S(table([
    ["Area", "Requirement", "How it is met"],
    ["Latency SLO", "TTFT p95 ≤ 500 ms; e2e p95 ≤ 3 s", "Recording rules + InferenceTTFTSLOViolation (30 s)"],
    ["Availability", "Error rate ≤ 2%; canary failures invisible to users",
     "Retry to a different backend before first token; breaker; fallback to stable"],
    ["Overload", "Degrade predictably; protect paying tenants", "Bulkhead + bounded priority queue; 429/503 with "
     "Retry-After"],
    ["Safety of change", "No production change without a human decision", "Write tools only via approval gate; "
     "RBAC operator role"],
    ["Auditability", "Who changed what, when, why", "audit table, deployment_events, stage_transitions, "
     "incident timeline"],
    ["Recoverability", "Agent runs survive restarts", "LangGraph SQLite checkpointer; resume by incident id"],
    ["Observability", "Every request attributable by version and tenant", "Labelled histograms/counters; trace "
     "spans; x-trace-id header"],
], [0.17, 0.38, 0.45]))
S(PageBreak())

S(P("2.7 Key design decisions", "h2"))
S(table([
    ["Decision", "Alternative considered", "Rationale"],
    ["Gateway polls control plane for routing", "Push / watch stream",
     "Polling is trivially robust to control-plane restarts; last-known-good table gives graceful degradation"],
    ["Metrics pull (Prometheus); OTLP only for traces + logs", "OTLP metrics push",
     "vLLM, DCGM and kube-state-metrics speak Prometheus; one TSDB for every layer"],
    ["Retry only before first token, to a different backend first", "Retry same backend; retry mid-stream",
     "Mid-stream retry duplicates tokens; retrying the failing canary wastes the latency budget"],
    ["Slow canary handled by agent, not breaker", "Trip breaker on latency",
     "Breaker trips on errors / TTFT timeout (outage); a 3× slowdown is a regression needing diagnosis"],
    ["Agent proposes; graph executes", "LLM calls write tools directly",
     "Contains prompt injection: worst case is a wrong recommendation shown to a human"],
    ["Separate propose and await_approval nodes", "One node",
     "LangGraph re-runs an interrupted node on resume; side effects before interrupt() would duplicate"],
    ["SQLite (WAL) state store", "Postgres", "Zero-ops for one replica; plain SQL keeps migration a "
     "connection-string change"],
    ["Simulator with a capacity model", "Echo/mock server",
     "Incidents emerge from load (queueing, KV pressure) instead of being scripted"],
    ["Ship gate as policy with budgets", "Leaderboard / single metric",
     "Quality-vs-latency-vs-cost is a decision; budgets make it explicit and reviewable"],
], [0.3, 0.22, 0.48]))

S(P("2.8 Security model", "h2"))
story.extend(bullets([
    "<b>Authentication:</b> bearer API keys mapped to (tenant, role). <b>Roles:</b> viewer &lt; operator &lt; "
    "admin. Inference and reads need viewer; any mutation, approval or webhook needs operator.",
    "<b>Guardrails</b> before any model sees input: message shape, prompt size cap, credential detection "
    "(AWS keys, private keys, API keys), max_tokens bounds.",
    "<b>Agent containment:</b> read-only investigation tools; tool output wrapped as untrusted data; the only "
    "LLM output channel is a strict <font name='Mono'>submit_diagnosis</font> schema whose action is an enum; the "
    "graph re-validates the action, strips unknown args and refuses cross-deployment targets.",
    "<b>Idempotent writes:</b> agent actions carry <font name='Mono'>incident_id:action</font> keys; API callers "
    "may send <font name='Mono'>Idempotency-Key</font>.",
    "<b>Network:</b> model servers are reachable only from gateway, control plane and monitoring (NetworkPolicy); "
    "containers run as non-root with read-only root filesystems.",
]))
S(PageBreak())

# ============================================================================= 3. LLD
S(P("3. Low-level design", "h1"))
S(P("3.1 Code layout", "h2"))
S(table([
    ["Module", "Lines", "Key types / functions"],
    ["forge/config.py", "96", "Settings (env-driven), parsers for API keys and tenant limits"],
    ["forge/db.py", "114", "SCHEMA, connect() (WAL), tx() (BEGIN IMMEDIATE), audit()"],
    ["forge/telemetry.py", "66", "setup(service, app): tracer provider, OTLP spans + logs, instrumentation"],
    ["forge/training/dataset.py · train.py", "69 · 232", "prepare(); TrainConfig, memory_plan(), train()"],
    ["forge/evaluation/harness.py · scorers.py · compare.py", "171 · 39 · 83",
     "run_eval() → EvalReport.headline(); exact_match/contains/token_f1; ShipPolicy, compare()"],
    ["forge/registry/registry.py", "160", "Registry.register/transition/update_metrics/history; ALLOWED"],
    ["forge/deploy/controller.py", "215", "DeployController.create/start_canary/set_weight/promote/rollback/"
     "pause/scale/routing; _mutate()"],
    ["forge/gateway/*.py", "~580", "app (chat_completions, Gateway.open_upstream), routing (RoutingTable.plan), "
     "admission (Bulkhead), breaker, ratelimit, retry, guardrails, auth, metrics"],
    ["forge/serving/simulator.py", "287", "Profile, Faults, Engine (acquire/release, kv_usage), completions()"],
    ["forge/agent/tools.py · prom.py · cost.py", "257 · 94 · 79", "Toolbox (17 tools), Prom/Tempo/Loki, "
     "explain_cost_change()"],
    ["forge/agent/investigator.py · graph.py · incidents.py", "326 · 178 · 64",
     "Heuristic/ClaudeInvestigator, Diagnosis; IncidentGraph; IncidentStore"],
    ["forge/controlplane/app.py · static/index.html", "317", "REST API, webhook, overview, UI"],
    ["forge/mcp_server.py · cli.py", "85 · 430", "MCPServer registration; Typer app"],
], [0.4, 0.1, 0.5]))

S(P("3.2 Data model (SQLite)", "h2"))
S(table([
    ["Table", "Columns", "Notes"],
    ["model_versions", "name, version (PK), base_model, adapter_uri, serving_url, dataset_hash, run_id, stage, "
     "metrics JSON, tags JSON, created_at", "Immutable except stage, metrics, serving_url"],
    ["stage_transitions", "id, name, version, from_stage, to_stage, actor, reason, at", "Append-only history"],
    ["deployments", "name (PK), model, stable_version, canary_version, canary_weight, status, revision, replicas, "
     "updated_at", "status ∈ stable | canary | paused"],
    ["deployment_events", "id, deployment, action, detail JSON, actor, revision, at",
     "Answers “what changed?” for the agent"],
    ["incidents", "id (PK), title, severity, state, source, labels JSON, diagnosis JSON, proposed_action JSON, "
     "timeline JSON, created_at, updated_at", "Projection of the graph run; labels.fingerprint de-duplicates alerts"],
    ["idempotency", "key (PK), response JSON, at", "Stored in the same transaction as the mutation"],
    ["audit", "id, actor, action, target, detail JSON, at", "Every API mutation and approval"],
], [0.18, 0.52, 0.3], mono_cols=(1,)))
S(P("LangGraph checkpoints live in a separate SQLite file (agent_checkpoints.db), keyed by thread_id = incident id.",
    "small"))

d = Drawing(W, 150)
G = (W - 4 * 88) / 3
X = [i * (88 + G) for i in range(4)]
pos = {"candidate": (X[0], 80), "staging": (X[1], 80), "production": (X[2], 80), "archived": (X[3], 80),
       "rejected": (X[1], 10)}
for k, (x, y) in pos.items():
    box(d, x, y, 88, 30, k, fill=C_BAD if k == "rejected" else (C_DP if k == "production" else C_CP),
        stroke=MUTED)
arrow(d, X[0] + 88, 95, X[1], 95, "canary", ly=4)
arrow(d, X[1] + 88, 95, X[2], 95, "promote", ly=4)
arrow(d, X[2] + 88, 95, X[3], 95, "superseded", ly=4)
poly_arrow(d, [(X[0] + 44, 110), (X[0] + 44, 132), (X[2] + 44, 132), (X[2] + 44, 110)], "direct promote",
           lpos=((X[0] + X[2]) / 2 + 44, 129))
poly_arrow(d, [(X[3] + 44, 80), (X[3] + 44, 60), (X[2] + 60, 60), (X[2] + 60, 80)], "rollback target",
           lpos=(X[3] + 4, 57))
arrow(d, X[1] + 44, 80, X[1] + 44, 40, "canary rolled back", lx=36)
poly_arrow(d, [(X[1], 22), (X[0] + 30, 22), (X[0] + 30, 80)], "re-qualify", lpos=((X[0] + 30 + X[1]) / 2, 19))
arrow(d, X[0] + 70, 80, X[1] + 10, 40, "reject", lx=-10, ly=6)
S(fig("3.3 Model registry lifecycle", d, "Figure 3. Stage transitions (Registry.ALLOWED). Exactly one production version per model: promoting "
          "archives the previous one in the same transaction. Rejected/archived versions cannot be canaried "
          "until re-qualified."))

S(PageBreak())

S(P("3.4 Deploy controller", "h2"))
S(P("Every mutation goes through <font name='Mono'>_mutate(name, op, actor, expected_revision, idempotency_key)"
    "</font>:"))
S(code("""1. if idempotency_key exists in idempotency table -> return stored response (no-op)
2. BEGIN IMMEDIATE                                   (take the write lock up front)
3. load deployment; if expected_revision != revision -> DeployError("revision conflict")
4. action, detail = op(conn, deployment)              (validates + UPDATEs)
5. revision += 1; updated_at = now
6. INSERT deployment_events(action, detail, actor, revision)
7. INSERT idempotency(key, response)                  (same transaction as the change)
8. COMMIT; then registry side effects (staging / production / rejected transitions)"""))
S(table([
    ["Operation", "Preconditions", "Effect"],
    ["create", "version has serving_url; name unused", "stable = version; registry → production"],
    ["start_canary", "not stable; not rejected/archived; no other canary; weight ∈ (0,100]",
     "canary = version, weight, status canary; candidate → staging"],
    ["set_weight", "canary exists; weight ∈ [0,100]", "canary_weight = weight"],
    ["pause", "status canary", "status paused (weights frozen)"],
    ["promote", "canary exists", "stable = canary; canary cleared; registry → production (old archived)"],
    ["rollback", "canary exists, or a previous stable is known", "abort canary (→ rejected) or revert stable to "
     "the version it replaced (from promote events)"],
    ["scale", "replicas ∈ [1,64]", "replicas recorded (Kubernetes/KEDA act on it)"],
], [0.15, 0.42, 0.43]))
S(P("<b>routing()</b> produces, per deployment: <font name='Mono'>{revision, status, backends: [{model, version, "
    "url, weight, role, price_per_1k_tokens}], fallback: stable}</font>. Stable weight is 100 − canary weight; a "
    "canary at weight 0 is omitted."))

S(P("3.5 Gateway request pipeline", "h2"))
d = Drawing(W, 96)
stages = [("Auth", "API key →\ntenant · role"), ("Rate limit", "token bucket\nper tenant"),
          ("Guardrails", "shape · size\ncredentials"), ("Route", "weighted pick\n+ sticky session"),
          ("Breaker", "skip open\nbackends"), ("Admission", "bulkhead +\npriority queue"),
          ("Upstream", "TTFT timeout\nretry · fallback"), ("Relay", "stream · TPOT\ntokens · cost")]
sw = (W - 7 * 6) / 8
for i, (t, s) in enumerate(stages):
    x = i * (sw + 6)
    box(d, x, 40, sw, 48, t, s, fill=C_DP, stroke=B_DP, fs=7.6, sub_fs=6.2)
    if i:
        arrow(d, x - 6, 64, x, 64)
rej = ["401", "429", "400/413", "404", "skip → next", "429 / 503", "503 if all fail", "error event"]
for i, r in enumerate(rej):
    x = i * (sw + 6) + sw / 2
    arrow(d, x, 40, x, 18, color=colors.HexColor("#b42318"), dashed=True)
    d.add(String(x, 8, r, fontName="Sans", fontSize=6.6, fillColor=colors.HexColor("#b42318"),
                 textAnchor="middle"))
S(d)
S(caption("Figure 4. Gateway stages in order; red dashed arrows are the rejection or failure outcome of each stage."))
S(PageBreak())

S(P("3.6 Gateway algorithms", "h2"))
S(P("Token bucket (per tenant)", "h3"))
S(P("tokens = min(burst, tokens + elapsed × rate); admit if tokens ≥ 1, else 429 with Retry-After = (1 − tokens) / "
    "rate. Limits come from <font name='Mono'>FORGE_TENANT_LIMITS</font> as rate/burst/priority, defaulting to "
    "platform 100/200/2, acme 20/40/1, free-tier 5/10/0."))
S(P("Admission control (bulkhead per backend)", "h3"))
S(P("Each backend (model:version) has a semaphore of <i>max_inflight</i> (32) and a bounded queue (64). A priority "
    "class may only occupy a share of the queue: <b>priority 0 → 50%</b>, <b>1 → 80%</b>, <b>2 → 100%</b>. When "
    "in-flight is full and the queue is beyond the caller's share, the request is shed with 429 (queue_full). A "
    "request that waits longer than <i>queue_timeout</i> (5 s) gets 503 (queue_timeout). Under pressure the free "
    "tier is shed first."))
S(P("Circuit breaker (per backend)", "h3"))
d = Drawing(W, 92)
box(d, 20, 30, 110, 32, "CLOSED", "record outcomes (window 50)", fill=C_DP, stroke=B_DP)
box(d, W / 2 - 55, 30, 110, 32, "OPEN", "reject; wait open_s (15 s)", fill=C_BAD, stroke=colors.HexColor("#b42318"))
box(d, W - 130, 30, 110, 32, "HALF_OPEN", "one probe at a time", fill=C_OBS, stroke=B_OBS)
arrow(d, 130, 52, W / 2 - 55, 52, "failure ratio ≥ 50% over ≥ 10 calls", ly=4)
arrow(d, W / 2 + 55, 52, W - 130, 52, "open_s elapsed", ly=4)
poly_arrow(d, [(W - 75, 30), (W - 75, 12), (75, 12), (75, 30)], "probe succeeds → clear window",
           lpos=(W / 2, 3))
poly_arrow(d, [(W - 75, 62), (W - 75, 82), (W / 2, 82), (W / 2, 62)], "probe fails", lpos=(W * 0.72, 85))
S(d)
S(P("Failures counted: connect errors, HTTP 5xx / 429 from the model server, TTFT timeout (10 s), and streams that "
    "break mid-response."))
S(P("Routing, retry and fallback", "h3"))
S(P("<font name='Mono'>plan(deployment, sticky_key, attempts)</font> picks a primary by weighted random (seeded by "
    "<font name='Mono'>x-session-id</font> for stickiness) and returns <b>[primary, other backends…, fallback, "
    "primary]</b> truncated to retries + 1 (3). Attempts skip backends whose breaker is open, sleep "
    "full-jitter backoff <font name='Mono'>uniform(0, min(2 s, 0.1 × 2^n))</font> between attempts, and are "
    "allowed only until the first content token has been received. After that, the response is committed: a "
    "broken stream emits an in-band error chunk and counts as a breaker failure. A request served by a "
    "non-primary backend increments <font name='Mono'>forge_fallback_total{from_version, reason}</font>."))
S(P("Streaming relay accounting", "h3"))
S(P("The gateway always requests streaming upstream (with <font name='Mono'>include_usage</font>), even for "
    "non-streaming clients, so TTFT is measured the same way for everyone. On completion it records: e2e duration, "
    "TPOT = (t_end − t_first) / (completion_tokens − 1), prompt and completion tokens per tenant, cost = tokens / "
    "1000 × price_per_1k_tokens of the version that served, and the status code. Response headers: "
    "<font name='Mono'>x-forge-deployment, x-forge-version, x-forge-route (stable|canary|fallback), x-trace-id</font>."))

S(P("3.7 Model-server simulator capacity model", "h2"))
S(code("""admission     : at most max_num_seqs * capacity_multiplier running; others wait  -> vllm:num_requests_waiting
kv_usage      : resident_tokens / kv_capacity_tokens                           -> vllm:gpu_cache_usage_perc
prefill (TTFT): (ttft_base + prefill_ms_per_1k * prompt_tokens/1000) * ttft_mult * (1 + 1.5 * kv_usage^2)
decode (TPOT) : tpot_ms * tpot_mult * (1 + batch_penalty * (running-1)) * (1 + kv_usage)
quality       : question rank (version-independent hash) < round(accuracy * N)  -> correct answer
GPU           : util ~ 90 * running/capacity ; FB used = mem_base + extra_mem + 6 GB * kv_usage"""))
S(P("v3 differs from v2 only in its profile (max_num_seqs 4 vs 16, KV capacity 1.8k tokens, +4.5 GB base memory). "
    "At evaluation concurrency 4 it never queues, so it passes the ship gate. At 50% of production traffic it "
    "queues, which is the regression the agent must find."))
S(PageBreak())

S(P("3.8 Evaluation and ship gate", "h2"))
S(P("<font name='Mono'>run_eval()</font> streams every eval prompt (bounded concurrency), measuring TTFT, latency and "
    "usage. Headline metrics written to the registry: accuracy (primary scorer, default <i>contains</i>), token_f1, "
    "ttft_p95_s, latency_p50/p95_s, tokens_per_s (decode), avg_output_tokens, cost_per_1k_tokens, cost_per_request, "
    "error_rate. Cost model: <font name='Mono'>$/1k tokens = (gpu_hourly_usd / 3600) / aggregate_tokens_per_s × "
    "1000</font>. A GPU costs the same per hour whether busy or idle, so cost per token is set by throughput."))
S(table([
    ["Rule (ShipPolicy defaults)", "Outcome"],
    ["Accuracy drops more than 1 pt, or error rate > 2%", "BLOCK"],
    ["p95 latency +>20%, TTFT p95 +>25%, or cost/request +>25%, and accuracy gain ≥ 2 pts", "NEEDS_REVIEW (trade-off "
     "stated in the reason)"],
    ["Same regressions with accuracy gain < 2 pts", "BLOCK"],
    ["Otherwise", "SHIP_CANARY"],
], [0.68, 0.32]))

S(P("3.9 Training", "h2"))
story.extend(bullets([
    "<b>Data:</b> accepts <font name='Mono'>messages</font> or <font name='Mono'>prompt/response</font> rows; "
    "validates roles and a final assistant turn; removes duplicates; deterministic seeded split; SHA-256 of the "
    "raw file recorded as lineage.",
    "<b>Loss masking:</b> each turn's text is the diff between the chat-template render before and after it. "
    "Role headers and system/user turns get label −100; assistant content plus its end-of-turn token are trained. "
    "Non-prefix-stable templates are rejected.",
    "<b>Loop:</b> AdamW on LoRA params, cosine schedule with warmup, gradient accumulation, grad-clip 1.0, "
    "gradient checkpointing, periodic val loss and checkpoints; QLoRA loads NF4 double-quantised weights.",
    "<b>Memory plan:</b> weights (2 B/param bf16, ~0.55 NF4) + LoRA params × 16 B (weights, grads, Adam m/v, fp32 "
    "master) + activations (batch × seq × hidden × layers × 2 with checkpointing, 34 without) + logits (batch × "
    "seq × vocab × 4 B) + 1 GB overhead. LoRA params use real projection shapes (GQA k/v, MLP intermediate size).",
]))

S(P("3.10 Agent", "h2"))
S(table([
    ["Tool", "Risk", "Source"],
    ["get_slo_status, get_service_health, get_model_metrics", "read", "Prometheus (gateway SLIs by deployment / "
     "version)"],
    ["get_gpu_metrics, get_queue_status, get_recent_errors", "read", "Prometheus (vLLM, DCGM, gateway), Loki"],
    ["get_deployment_status, compare_model_versions", "read", "Controller state + events; registry eval metrics"],
    ["search_slow_traces, get_trace", "read", "Tempo"],
    ["explain_cost_change", "read", "Prometheus increase() over window vs offset window"],
    ["rollback_model, pause_rollout, set_canary_weight, start_canary, promote_canary, scale_inference", "write",
     "DeployController via approval gate only"],
], [0.5, 0.08, 0.42], mono_cols=(0,)))
S(P("Investigators", "h3"))
story.extend(bullets([
    "<b>ClaudeInvestigator</b> (claude-opus-5-5, adaptive thinking, effort high, server-side refusal fallback): a "
    "manual tool-use loop over read tools only, at most 12 turns, ending in a strict "
    "<font name='Mono'>submit_diagnosis</font> call (action ∈ rollback_model, pause_rollout, set_canary_weight, "
    "scale_inference, none).",
    "<b>HeuristicInvestigator</b> (runbook): SLO → what changed → suspect vs baseline (TTFT ratio > 1.5 or error "
    "rate > max(2× baseline, SLO)) → rollback; else saturation (queue or GPU > 85%) → scale +1; else escalate. Used "
    "when Claude is unavailable, and as the fallback when the LLM path errors, refuses or never submits.",
]))
S(PageBreak())

S(P("Incident state machine", "h3"))
d = Drawing(W, 205)
nodes = {"INCIDENT": (0, 155), "INVESTIGATING": (1, 155), "DIAGNOSIS_READY": (2, 155),
         "AWAITING_APPROVAL": (3, 155), "EXECUTING": (3, 85), "VERIFYING": (2, 85), "RESOLVED": (1, 85),
         "ESCALATED": (0, 10), "VERIFY_FAILED": (2, 10), "REJECTED": (3, 10)}
nw = (W - 3 * 18) / 4
coords = {}
for k, (cx_, cy_) in nodes.items():
    x = cx_ * (nw + 18)
    coords[k] = (x, cy_)
    fill = C_DP if k == "RESOLVED" else (C_BAD if k in ("ESCALATED", "REJECTED", "VERIFY_FAILED") else
                                         (C_AG if k == "AWAITING_APPROVAL" else C_CP))
    box(d, x, cy_, nw, 30, k, fill=fill, fs=7.4)


def hedge(a, b, text, below=False):
    (ax, ay), (bx, by) = coords[a], coords[b]
    if bx > ax:
        arrow(d, ax + nw, ay + 15, bx, by + 15)
    else:
        arrow(d, ax, ay + 15, bx + nw, by + 15)
    label(d, (min(ax, bx) + nw + max(ax, bx)) / 2, ay - 10 if below else ay + 34, text, color=MUTED)


hedge("INCIDENT", "INVESTIGATING", "intake")
hedge("INVESTIGATING", "DIAGNOSIS_READY", "investigate")
hedge("DIAGNOSIS_READY", "AWAITING_APPROVAL", "propose")
hedge("EXECUTING", "VERIFYING", "execute")
hedge("VERIFYING", "RESOLVED", "3 healthy checks", below=True)
ax_ = coords["AWAITING_APPROVAL"][0] + nw / 2
arrow(d, ax_ - 20, 155, ax_ - 20, 115)
label(d, ax_ - 16, 132, "approved", color=MUTED, anchor="start")
poly_arrow(d, [(ax_ + 30, 155), (ax_ + 30, 140), (W - 4, 140), (W - 4, 40)])
label(d, W - 8, 62, "rejected", color=MUTED, anchor="end")
dx = coords["DIAGNOSIS_READY"][0]
poly_arrow(d, [(dx + 10, 155), (dx + 10, 135), (nw / 2, 135), (nw / 2, 40)])
label(d, nw / 2 + 4, 90, "action = none", color=MUTED, anchor="start")
vx = coords["VERIFYING"][0] + nw / 2
arrow(d, vx, 85, vx, 40)
label(d, vx + 4, 60, "window expired", color=MUTED, anchor="start")
d.add(String(coords["AWAITING_APPROVAL"][0] + nw / 2, 196, "interrupt()  ·  durable checkpoint", fontName="Sans",
             fontSize=6.6, fillColor=B_AG, textAnchor="middle"))
S(d)
S(caption("Figure 5. Graph nodes: intake → investigate → propose → await_approval (interrupt) → execute → verify; "
          "escalate when no safe action exists. Every transition is mirrored to the incidents table."))
S(table([
    ["IncidentState field", "Content"],
    ["incident_id, deployment, alert", "Thread key; target deployment; alert labels/annotations (untrusted)"],
    ["phase", "Current state name"],
    ["diagnosis", "summary, root_cause, evidence[], confidence, suspected_change, action, action_args, rationale, "
     "investigator, tool_calls[]"],
    ["decision", "{approved, by, comment} supplied on resume"],
    ["action_result, verification", "Controller response; last SLO snapshot and ok flag"],
    ["log", "Append-only (operator.add reducer) node log"],
], [0.3, 0.7], mono_cols=(0,)))
S(P("Execute-time checks", "h3"))
story.extend(bullets([
    "Action must be in the allowlist and be a <i>write</i> tool; <font name='Mono'>none</font> never reaches execute.",
    "<font name='Mono'>action_args.deployment</font> must equal the incident's deployment; unknown args are dropped.",
    "Idempotency key <font name='Mono'>{incident_id}:{action}</font>; actor recorded as "
    "<font name='Mono'>agent/approved-by:{who}</font>.",
    "Verify: poll SLOs every 5 s within FORGE_VERIFY_WINDOW_S (150 s in compose); 3 consecutive in-SLO checks → "
    "RESOLVED.",
]))
S(PageBreak())

S(P("3.11 APIs", "h2"))
S(table([
    ["Method & path", "Role", "Purpose"],
    ["GET /api/models  ·  POST /api/models", "viewer · operator", "List / register versions"],
    ["POST /api/models/{n}/{v}/metrics | /stage | /serving-url", "operator", "Record eval metrics; transition; "
     "set endpoint"],
    ["GET /api/models/{n}/history  ·  /compare?base&candidate", "viewer", "Stage history; ship-gate comparison"],
    ["GET /api/routing", "viewer", "Gateway routing table"],
    ["GET|POST /api/deployments  ·  GET /{name}/events", "viewer · operator", "List / create; event log"],
    ["POST /api/deployments/{name}/{canary|weight|promote|rollback|pause|scale}", "operator",
     "Rollout operations; Idempotency-Key header; expected_revision in body"],
    ["GET /api/incidents[/{id}]  ·  POST /api/incidents", "viewer · operator", "List / detail with graph state; "
     "open (optional proposal)"],
    ["POST /api/incidents/{id}/decision", "operator", "Approve / reject; resumes the graph"],
    ["POST /webhooks/alertmanager", "operator", "Firing alerts → de-duplicated incidents"],
    ["POST /api/agent/ask  ·  GET /api/overview", "viewer", "Free-form question; UI data"],
    ["Gateway: POST /v1/chat/completions  ·  GET /v1/models", "viewer", "OpenAI-compatible inference"],
    ["Model server: /v1/chat/completions, /v1/models, /health, /metrics, /admin/faults", "(internal)",
     "vLLM-compatible; faults only on the simulator"],
    ["/healthz, /metrics on every service", "none", "Probes and scraping"],
], [0.5, 0.15, 0.35], mono_cols=(0,)))

S(P("3.12 Metrics and alerts", "h2"))
S(table([
    ["Metric", "Type · labels"],
    ["forge_requests_total", "counter · deployment, version, tenant, code"],
    ["forge_request_duration_seconds, forge_ttft_seconds, forge_tpot_seconds", "histogram · deployment, version"],
    ["forge_tokens_total, forge_cost_usd_total", "counter · deployment, version, tenant (+ kind)"],
    ["forge_inflight_requests, forge_queue_depth, forge_breaker_state", "gauge · backend"],
    ["forge_rejected_total, forge_fallback_total, forge_retries_total, forge_upstream_errors_total",
     "counter · reason / from_version / backend, kind"],
    ["forge_routing_revision, forge_canary_weight, forge_routing_stale_seconds", "gauge"],
    ["vllm:num_requests_running|waiting, vllm:gpu_cache_usage_perc, vllm:time_to_first_token_seconds, …",
     "model server (vLLM names)"],
    ["DCGM_FI_DEV_GPU_UTIL, DCGM_FI_DEV_FB_USED, DCGM_FI_DEV_FB_FREE", "GPU (DCGM exporter names)"],
], [0.62, 0.38], mono_cols=(0,)))
S(Spacer(1, 6))
S(table([
    ["Alert", "Condition", "Severity → route"],
    ["InferenceTTFTSLOViolation", "forge:ttft_p95:1m > 0.5 and traffic > 0.2 rps, for 30 s", "critical → agent"],
    ["InferenceErrorRateHigh", "forge:error_ratio:1m > 2%, for 1 m", "critical → agent"],
    ["InferenceQueueBacklog", "queue depth > 10, for 1 m", "warning"],
    ["CircuitBreakerOpen", "breaker state = 2, for 15 s", "warning"],
    ["GatewayRoutingStale", "routing stale > 30 s, for 1 m", "warning"],
    ["GPUMemoryPressure", "FB used / total > 90%, for 2 m", "warning"],
], [0.3, 0.5, 0.2], mono_cols=(0,)))
# ============================================================================= 4. FLOWS
S(P("4. Flows", "h1"))
S(P("Sequence diagrams for the main paths. Solid arrows are calls, dashed arrows are responses, red arrows are "
    "failures, yellow boxes are notes."))
CM = {"cli": C_ST, "cp": C_CP, "reg": C_CP, "ctl": C_CP, "gw": C_DP, "ms": C_DP, "ms2": C_DP, "v2": C_DP,
      "v3": C_DP, "prom": C_OBS, "am": C_OBS, "graph": C_AG, "agent": C_AG, "tools": C_AG, "human": C_ST,
      "client": C_ST, "mcp": C_ST, "train": C_CP, "hf": C_ST}

S(P("F1. Model lifecycle: dataset to production", "h2"))
S(sequence(
    [("cli", "ML engineer\nforge CLI"), ("train", "Training"), ("hf", "HF Hub"), ("reg", "Registry"),
     ("ms", "Model server\nvN (vLLM)"), ("ctl", "Deploy\ncontroller")],
    [("cli", "train", "model train --config", "solid"),
     ("train", "train", "validate · dedupe · hash dataset", "self"),
     ("train", "hf", "download base model", "solid"),
     ("train", "train", "LoRA/QLoRA SFT · val loss", "self"),
     ("train", "cli", "run_id · adapter/ · manifest.json", "reply"),
     ("cli", "reg", "model register --run (lineage)", "solid"),
     ("reg", "cli", "stage = candidate", "reply"),
     ("cli", "ms", "vllm serve --lora-modules name-vN=adapter", "solid"),
     ("cli", "ms", "model evaluate: stream eval set", "solid"),
     ("ms", "cli", "TTFT · latency · tokens · answers", "reply"),
     ("cli", "reg", "headline metrics", "solid"),
     ("cli", "reg", "model compare base vN → ship gate", "solid"),
     ("reg", "cli", "SHIP_CANARY | NEEDS_REVIEW | BLOCK", "reply"),
     ("cli", "ctl", "model deploy vN --weight 10", "solid"),
     ("ctl", "reg", "candidate → staging", "solid"),
     ("cli", "ctl", "model weight 50 … model promote", "solid"),
     ("ctl", "reg", "→ production (previous → archived)", "solid")],
    colors_map=CM))
S(caption("Figure 6. Train → register → evaluate → gate → canary → promote. BLOCK stops the flow; NEEDS_REVIEW "
          "requires an explicit human trade-off decision."))

S(P("F2. Inference request (happy path, streaming)", "h2"))
S(sequence(
    [("client", "Tenant app"), ("gw", "Gateway"), ("cp", "Control plane"), ("ms", "Model server\n(chosen)"),
     ("prom", "Prometheus")],
    [("gw", "cp", "GET /api/routing (every 2 s, background)", "solid"),
     ("cp", "gw", "routes rev N (cached: last-known-good)", "reply"),
     ("client", "gw", "POST /v1/chat/completions stream=true", "solid"),
     ("gw", "gw", "auth · token bucket · guardrails", "self"),
     ("gw", "gw", "plan(): weighted pick (sticky by session)", "self"),
     ("gw", "gw", "breaker.allow() · bulkhead.admit(priority)", "self"),
     ("gw", "ms", "stream=true, include_usage", "solid"),
     ("ms", "ms", "queue for batch slot → prefill", "self"),
     ("ms", "gw", "first token (TTFT measured)", "reply"),
     ("gw", "client", "200 + x-forge-version / route / trace-id", "reply"),
     ("ms", "gw", "token chunks …", "reply"),
     ("gw", "client", "SSE chunks …", "reply"),
     ("ms", "gw", "final chunk + usage", "reply"),
     ("gw", "client", "data: [DONE]", "reply"),
     ("gw", "gw", "record TTFT · TPOT · tokens · cost · breaker ok", "self"),
     ("prom", "gw", "scrape /metrics (5 s)", "solid")],
    colors_map=CM))
S(caption("Figure 7. The control plane is not on the request path; the gateway serves from its cached table."))

S(P("F3. Canary failure: retry elsewhere, fallback, breaker", "h2"))
S(sequence(
    [("client", "Tenant app"), ("gw", "Gateway"), ("v3", "Canary v3"), ("v2", "Stable v2")],
    [("client", "gw", "request", "solid"),
     ("gw", "gw", "plan() = [v3, v2, v3]", "self"),
     ("gw", "v3", "attempt 1", "solid"),
     ("v3", "gw", "HTTP 500 / no token within 10 s", "bad"),
     ("gw", "gw", "breaker(v3).record(fail) · backoff (jitter)", "self"),
     ("gw", "v2", "attempt 2 (different backend)", "solid"),
     ("v2", "gw", "first token", "reply"),
     ("gw", "client", "200, x-forge-route: fallback", "reply"),
     ("gw", "gw", "forge_fallback_total{from_version=v3}++", "self"),
     ("gw", "gw", "after ≥50% failures in window: breaker(v3) OPEN", "note"),
     ("client", "gw", "next request", "solid"),
     ("gw", "gw", "v3 skipped (open) → v2 directly", "self"),
     ("gw", "v2", "attempt", "solid"),
     ("v2", "gw", "stream", "reply"),
     ("gw", "gw", "after 15 s: HALF_OPEN, one probe to v3", "note")],
    colors_map=CM))
S(caption("Figure 8. Users never see the canary's failures: verified by the integration test (200 requests, zero "
          "non-200 responses while v3 returns 500)."))

S(P("F4. Incident: detection to verified recovery", "h2"))
S(sequence(
    [("prom", "Prometheus"), ("am", "Alertmanager"), ("cp", "Control plane"), ("graph", "Incident graph\n"
     "(LangGraph)"), ("tools", "Toolbox"), ("human", "On-call"), ("ctl", "Deploy\ncontroller")],
    [("prom", "prom", "ttft_p95 > 0.5 s for 30 s → firing", "self"),
     ("prom", "am", "InferenceTTFTSLOViolation", "solid"),
     ("am", "cp", "POST /webhooks/alertmanager", "solid"),
     ("cp", "cp", "dedupe by fingerprint · create incident", "self"),
     ("cp", "graph", "start(incident_id) [thread pool]", "solid"),
     ("graph", "tools", "get_slo_status · get_deployment_status", "solid"),
     ("graph", "tools", "service_health · model_metrics · gpu · queue · errors", "solid"),
     ("tools", "prom", "PromQL per version", "solid"),
     ("tools", "graph", "evidence (untrusted data)", "reply"),
     ("graph", "graph", "diagnosis: v3 regressed; action rollback_model", "self"),
     ("graph", "cp", "propose → AWAITING_APPROVAL; interrupt()", "solid"),
     ("graph", "graph", "checkpoint saved: run suspended", "note"),
     ("human", "cp", "forge incident approve <id> / UI button", "solid"),
     ("cp", "graph", "resume(Command(approved, by))", "solid"),
     ("graph", "graph", "validate action · same deployment · strip args", "self"),
     ("graph", "ctl", "rollback (idempotency key incident:action)", "solid"),
     ("ctl", "graph", "canary cleared · v3 → rejected · rev+1", "reply"),
     ("graph", "tools", "get_slo_status every 5 s", "solid"),
     ("graph", "cp", "3 healthy checks → RESOLVED", "solid")],
    colors_map=CM, step=18))
S(caption("Figure 9. Measured on the local stack: alert to diagnosis in seconds, approval to verified recovery "
          "~2 minutes (bounded by the 1-minute rate window draining)."))

S(P("F5. Cost attribution: “why did inference cost go up?”", "h2"))
S(sequence(
    [("human", "Operator"), ("cp", "Control plane"), ("agent", "Investigator"), ("tools", "Toolbox"),
     ("prom", "Prometheus")],
    [("human", "cp", "POST /api/agent/ask “why did cost increase…”", "solid"),
     ("cp", "agent", "ask(question)", "solid"),
     ("agent", "tools", "explain_cost_change(window, offset)", "solid"),
     ("tools", "prom", "increase(cost, requests, tokens{kind}) now", "solid"),
     ("tools", "prom", "same with offset (baseline window)", "solid"),
     ("tools", "prom", "per-version traffic mix + completion tokens", "solid"),
     ("tools", "tools", "C = R × T × P; ln(C1/C0) = Σ ln(factor ratios)", "self"),
     ("tools", "tools", "share of change per factor → primary driver", "self"),
     ("tools", "agent", "factors %, shares, tokens/request before→after", "reply"),
     ("agent", "cp", "narrative answer citing numbers", "reply"),
     ("cp", "human", "“tokens/request +63% (134→227) … 106% of change”", "reply")],
    colors_map=CM))
S(caption("Figure 10. Log decomposition makes factor contributions exactly additive; plain percentage deltas do "
          "not add up once changes are large."))

S(P("F6. External agent via MCP: same approval gate", "h2"))
S(sequence(
    [("mcp", "Claude Code\n(MCP client)"), ("cli", "forge MCP\nserver"), ("cp", "Control plane"),
     ("graph", "Incident graph"), ("human", "On-call")],
    [("mcp", "cli", "get_slo_status / get_model_metrics …", "solid"),
     ("cli", "mcp", "JSON (read-only tools)", "reply"),
     ("mcp", "cli", "request_action(rollback_model, reason)", "solid"),
     ("cli", "cp", "POST /api/incidents {proposal}", "solid"),
     ("cp", "cp", "proposal must target incident deployment", "self"),
     ("cp", "graph", "start(…, proposal) → skips investigate", "solid"),
     ("graph", "cp", "AWAITING_APPROVAL · interrupt()", "solid"),
     ("cli", "mcp", "incident_id · “awaiting human approval”", "reply"),
     ("human", "cp", "approve / reject", "solid"),
     ("cp", "graph", "resume → execute → verify (as F4)", "solid")],
    colors_map=CM))
S(caption("Figure 11. No MCP tool can mutate production directly; one HITL policy covers every agent."))

S(P("F7. Control-plane outage: graceful degradation", "h2"))
S(sequence(
    [("client", "Tenant app"), ("gw", "Gateway"), ("cp", "Control plane"), ("ms", "Model servers"),
     ("prom", "Prometheus")],
    [("gw", "cp", "GET /api/routing", "solid"),
     ("cp", "gw", "connection refused", "bad"),
     ("gw", "gw", "keep last-known-good table (rev N)", "self"),
     ("client", "gw", "requests", "solid"),
     ("gw", "ms", "served with rev N weights", "solid"),
     ("ms", "gw", "streams", "reply"),
     ("gw", "client", "200", "reply"),
     ("prom", "gw", "forge_routing_stale_seconds rising", "solid"),
     ("prom", "prom", "> 30 s for 1 m → GatewayRoutingStale (warning)", "self"),
     ("gw", "cp", "poll succeeds after recovery", "solid"),
     ("cp", "gw", "rev N+k → converge", "reply")],
    colors_map=CM))
S(caption("Figure 12. Rollout changes pause during the outage; traffic does not. On a cold start with no table, "
          "the gateway can load FORGE_ROUTING_FILE."))

S(P("F8. Overload: priority-aware load shedding", "h2"))
S(sequence(
    [("client", "Free-tier app\n(priority 0)"), ("client2", "Paying app\n(priority 2)"), ("gw", "Gateway "
     "bulkhead\n(inflight 32, queue 64)"), ("ms", "Model server")],
    [("gw", "gw", "inflight = 32/32 · queue = 40/64", "note"),
     ("client", "gw", "request", "solid"),
     ("gw", "client", "429 queue_full, Retry-After (share 50% = 32 exceeded)", "bad"),
     ("client2", "gw", "request", "solid"),
     ("gw", "gw", "queued (share 100% = 64 allows it)", "self"),
     ("gw", "ms", "dispatched when a slot frees", "solid"),
     ("ms", "gw", "stream", "reply"),
     ("gw", "client2", "200", "reply"),
     ("gw", "gw", "waited > 5 s → 503 queue_timeout instead", "note")],
    colors_map={**CM, "client2": C_ST}))
S(caption("Figure 13. Backpressure is explicit (429/503 with Retry-After) and ordered by tenant priority."))

S(PageBreak())
# ============================================================================= 5. VERIFICATION
S(P("5. Verification status and known limits", "h1"))
S(P("5.1 Verified", "h2"))
S(table([
    ["What", "How", "Result"],
    ["Registry, controller, gateway primitives, ship gate, cost attribution, dataset prep", "Unit tests",
     "Pass"],
    ["Canary split, streaming, auth, guardrails, fallback, breaker, metrics", "Integration test: real simulator + "
     "gateway processes", "Pass; 0 user-visible errors with canary at 100% 5xx"],
    ["Incident lifecycle incl. resume from checkpoint in a new process, rejection, cross-deployment refusal",
     "Agent tests (fake Prometheus)", "Pass"],
    ["Alert → diagnosis → approval → rollback → verify", "Docker stack, real Prometheus/Alertmanager, run twice",
     "RESOLVED both times"],
    ["Cost attribution on live data", "Output tokens ×1.9 on production for 5 min",
     "Driver correctly named: tokens/request"],
    ["Training loop, loss masking, memory planner", "CPU run on SmolLM2-135M; label decode; PEFT param count",
     "Adapter saved; masking exact; 4.88M = 4.88M"],
    ["Helm chart", "helm lint + template (16 resources)", "Pass"],
], [0.42, 0.33, 0.25]))
S(P("5.2 Not yet verified", "h2"))
story.extend(bullets([
    "Claude investigator against the live API (no key in the build environment); the runbook path was used.",
    "Real vLLM on GPU; QLoRA (requires CUDA + bitsandbytes); MLflow mirroring; Helm install on a cluster.",
]))
S(P("5.3 Known limits and next steps", "h2"))
S(table([
    ["Limit", "Impact", "Next step"],
    ["SQLite state store", "Single control-plane replica", "Postgres; same SQL"],
    ["In-process rate limits", "Limits are per gateway replica", "Redis-backed buckets behind the same interface"],
    ["Controller manages routing, not pods", "Model servers deployed separately", "Operator or GitOps (Argo CD) "
     "reconciling modelServers from the registry"],
    ["Reference-match eval scoring", "Weak signal for open-ended answers", "LLM-as-judge / pairwise scorers in "
     "SCORERS"],
    ["Agent has no cross-incident memory", "Repeats investigation work", "Store past diagnoses; retrieve by "
     "deployment + symptom"],
    ["Single-region", "No regional failover", "Per-region gateways with global routing"],
], [0.3, 0.3, 0.4]))

# Keep [heading] + drawing + caption on one page.
grouped, i = [], 0
while i < len(story):
    item = story[i]
    is_head = isinstance(item, Paragraph) and item.style.name in ("h2", "h3")
    if is_head and i + 1 < len(story) and isinstance(story[i + 1], Drawing):
        grp = [item, story[i + 1]]
        i += 2
    elif isinstance(item, Drawing):
        grp = [item]
        i += 1
    else:
        grouped.append(item)
        i += 1
        continue
    if i < len(story) and isinstance(story[i], Paragraph) and story[i].style.name == "caption":
        grp.append(story[i])
        i += 1
    grouped.append(KeepTogether(grp))
story[:] = grouped

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm,
                        bottomMargin=18 * mm, title="ForgeAI: System Design", author="ForgeAI",
                        subject="HLD, LLD and flows")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
