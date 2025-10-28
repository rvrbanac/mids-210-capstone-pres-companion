#!/usr/bin/env python3
"""
llm_consumer.py — Hybrid (Explicit + Semantic + LLM Reasoning) Slide Control
with multimodal semantic routing (MPNet + CLIP).
"""

import sys, time, re, json, os, pathlib, collections, requests, numpy as np, signal
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

# --- NEW: CLIP multimodal ---
from PIL import Image
try:
    import pillow_avif  # noqa: F401  (enable .avif support if installed)
except Exception:
    pass
try:
    from sentence_transformers import SentenceTransformer
    _CLIP_AVAILABLE = True
except Exception:
    _CLIP_AVAILABLE = False

# prevent broken pipe errors when piping output
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# ====================== CONFIG ======================
SERVER_URL       = "http://127.0.0.1:8765/command"
HTML_MODE        = True
HTML_DECK_PATH   = "/Users/rossvrbanac/MIDS/Capstone/modes_of_transit_visual.html"

# Text embeddings (for text→text)
EMBED_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
EMBED_CACHE_PATH = "/Users/rossvrbanac/MIDS/Capstone/slide_embeddings.json"

# CLIP model (text↔image joint space)
CLIP_MODEL_NAME  = os.getenv("CLIP_MODEL", "clip-ViT-B-32")  # works well + light

# Routing knobs
ENABLE_QA        = False
ENABLE_REASONING = True
MODEL_PATH       = '/Users/rossvrbanac/MIDS/267Summer25/Assignment 5/Mistral-7B-Instruct-v0.3.fp16.gguf'

ROLLING_SECONDS  = 45
CMD_DEBOUNCE_S   = 0.75
MIN_GAP_S        = 0.5
COALESCE_S       = 0.7

# --- Thresholds / mixing ---
SEM_STEADY_SEC    = 0.6
SEM_MIN_GAP_SEC   = 0.5

# weights for similarity blending; must sum to <= 1.0 (unused weight is ignored)
TEXT_SIM_WEIGHT   = float(os.getenv("TEXT_WEIGHT", 0.5))  # MPNet text→text
CLIP_SIM_WEIGHT   = float(os.getenv("CLIP_WEIGHT", 0.5))  # CLIP text→image/text

REASON_CHECK_GAP  = 2.0
REASON_WINDOW_S   = 45
REASON_MIN_CONF   = 0.15
REASON_STEADY_SEC = 2.0
LLM_MAX_TOKENS    = 128
LLM_TEMP          = 0.2
# ====================================================

# LLM load (optional reasoning)
try:
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=1000,
        n_threads=16,
        n_batch=2048,
        temperature=LLM_TEMP,
        top_p=0.95,
        max_tokens=LLM_MAX_TOKENS,
        repeat_penalty=1.1,
        verbose=False,
    )
except Exception as e:
    print(f"[warn] LLM not initialized: {e}", file=sys.stderr)
    ENABLE_REASONING = False

# Embeddings backends
base_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

if _CLIP_AVAILABLE:
    try:
        clip_model = SentenceTransformer(CLIP_MODEL_NAME)  # encodes texts & PIL images
    except Exception as e:
        print(f"[warn] Failed to init CLIP ({CLIP_MODEL_NAME}): {e}", file=sys.stderr)
        _CLIP_AVAILABLE = False

Q_PAT    = re.compile(r"\?|^(can|could|would|should|what|why|how|when|where|which|who|explain|tell me)\b", re.I)
NEXT_PAT = re.compile(r"\b(next\s+slide|next please|go\s*next|move\s+on)\b", re.I)
PREV_PAT = re.compile(r"\b(previous\s+slide|go\s*back|back\s+one)\b", re.I)
GOTO_PAT = re.compile(r"\bslide\s*(\d{1,3})\b", re.I)

# ---------- State ----------
buf = collections.deque()
last_call = 0.0
last_print = ""
CURRENT_SLIDE = 1
_last_cmd_sent = (None, 0.0)
_last_sem_check = 0.0
_best_idx_stable = None
_best_idx_since  = 0.0
_last_reason_check = 0.0
_reason_pick_stable = None
_reason_pick_since  = 0.0

def now(): 
    return time.time()

def prune():
    cutoff = now() - ROLLING_SECONDS
    while buf and buf[0][0] < cutoff:
        buf.popleft()

def recent_context(seconds=10):
    cutoff = now() - seconds
    return " ".join(t for ts, t in buf if ts >= cutoff)

# ---------- Helpers ----------
def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-9)
    return (v / n).astype(np.float32)

def cosine(a, b) -> float:
    return float(np.dot(a, b))

def safe_open_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[img-warn] open failed for {path}: {e}", file=sys.stderr)
        return None

# ---------- Slide catalog (now includes hidden labels, alt and filenames) ----------
def build_slide_catalog_from_html(path):
    base_dir = pathlib.Path(path).parent
    try:
        html = pathlib.Path(path).read_text(encoding="utf-8")
    except Exception as e:
        print(f"[error] Unable to read deck HTML at {path}: {e}", file=sys.stderr)
        return []
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select("section.slide") or soup.select("section") or soup.select("div.slide")
    slides = []
    for idx, sec in enumerate(nodes, start=1):
        title_el = sec.find(["h1", "h2"])
        title = title_el.get_text(" ", strip=True) if title_el else f"Slide {idx}"
        bullets = " ".join(li.get_text(" ", strip=True) for li in sec.find_all("li"))
        paras = " ".join(p.get_text(" ", strip=True) for p in sec.find_all("p")[:3])
        # hidden labels (e.g., <div style="display:none">motorcycle bike</div>)
        hidden = " ".join(d.get_text(" ", strip=True) for d in sec.find_all(True) if d.has_attr("style") and "display:none" in d["style"])

        # image info
        img_el = sec.find("img")
        img_rel = img_el["src"] if img_el and img_el.has_attr("src") else None
        img_alt = img_el.get("alt", "") if img_el else ""
        img_name = pathlib.Path(img_rel).stem if img_rel else ""
        img_path = str((base_dir / img_rel).resolve()) if img_rel else None

        text_blob = "\n".join(filter(None, [title, bullets, paras, hidden, img_alt, img_name])).strip()

        slides.append({
            "idx": idx,
            "title": title,
            "text": text_blob,
            "img_path": img_path
        })
    return slides

SLIDES = build_slide_catalog_from_html(HTML_DECK_PATH)

if not SLIDES:
    print(f"[error] No slides found in {HTML_DECK_PATH}. Semantic routing disabled.", file=sys.stderr)

# ---------- Embedding builders & cache ----------
def embed_text_mpnet(text: str) -> np.ndarray:
    v = np.array(base_embeddings.embed_query(text), dtype=np.float32)
    return l2norm(v)

def embed_text_clip(text: str) -> np.ndarray:
    if not _CLIP_AVAILABLE:
        return None
    v = clip_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.astype(np.float32)

def embed_image_clip(img_path: str) -> np.ndarray:
    if not (_CLIP_AVAILABLE and img_path and os.path.exists(img_path)):
        return None
    img = safe_open_image(img_path)
    if img is None:
        return None
    v = clip_model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.astype(np.float32)

def load_or_build_slide_embeddings(slides, cache_path):
    if not slides:
        return {
            "mpnet_text": [],
            "clip_text": [],
            "clip_image": [],
        }
    cache_ok = False
    payload = None
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if (payload.get("mpnet_model")==EMBED_MODEL_NAME and
                payload.get("clip_model")== (CLIP_MODEL_NAME if _CLIP_AVAILABLE else "none") and
                payload.get("num_slides")==len(slides)):
                cache_ok = True
    except Exception as e:
        print(f"[warn] Failed to read embedding cache, rebuilding: {e}", file=sys.stderr)

    if cache_ok:
        # restore
        mp = [np.array(v, dtype=np.float32) for v in payload.get("mpnet_text", [])]
        ct = [np.array(v, dtype=np.float32) for v in payload.get("clip_text", [])]
        ci = [np.array(v, dtype=np.float32) for v in payload.get("clip_image", [])]
        return {"mpnet_text": mp, "clip_text": ct, "clip_image": ci}

    # build fresh
    mpnet_text, clip_text, clip_image = [], [], []
    for s in slides:
        # MPNet on text
        mpnet_text.append(embed_text_mpnet(s["text"] or ""))

        # CLIP text (optional; helps when there IS text)
        ct = embed_text_clip(s["text"] or "") if _CLIP_AVAILABLE else None
        clip_text.append(ct if ct is not None else np.zeros(1, dtype=np.float32))  # placeholder

        # CLIP image (strong signal for visual-only slides)
        ci = embed_image_clip(s.get("img_path")) if _CLIP_AVAILABLE else None
        clip_image.append(ci if ci is not None else np.zeros(1, dtype=np.float32))  # placeholder

    # write cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "mpnet_model": EMBED_MODEL_NAME,
                "clip_model": (CLIP_MODEL_NAME if _CLIP_AVAILABLE else "none"),
                "num_slides": len(slides),
                "mpnet_text": [v.tolist() for v in mpnet_text],
                "clip_text": [v.tolist() if v.ndim>0 else [] for v in clip_text],
                "clip_image": [v.tolist() if v.ndim>0 else [] for v in clip_image],
            }, f)
    except Exception as e:
        print(f"[warn] Could not write embedding cache: {e}", file=sys.stderr)

    return {"mpnet_text": mpnet_text, "clip_text": clip_text, "clip_image": clip_image}

EMB = load_or_build_slide_embeddings(SLIDES, EMBED_CACHE_PATH)

# ---------- Slide commands ----------
def send_slide_cmd(kind, index=None):
    global _last_cmd_sent
    now_ts = now()
    last_kind, last_ts = _last_cmd_sent
    same = last_kind == (kind if kind != "GOTO" else ("GOTO", index))
    if same and (now_ts - last_ts) < CMD_DEBOUNCE_S:
        return
    payload = {"cmd": "next" if kind=="NEXT" else "prev" if kind=="PREV" else "goto"}
    if kind=="GOTO" and isinstance(index, int):
        payload["index"] = index
    print(f"[DEBUG] Sending {payload} -> {SERVER_URL}", flush=True)
    try:
        requests.post(SERVER_URL, json=payload, timeout=0.3)
    except Exception as e:
        print(f"LLM: (warn) slide server unreachable: {e}", flush=True)
    _last_cmd_sent = ((kind if kind!="GOTO" else ("GOTO", index)), now_ts)

def goto_slide(i):
    global CURRENT_SLIDE
    i = max(1, min(len(SLIDES), int(i)))
    send_slide_cmd("GOTO", i)
    CURRENT_SLIDE = i

def maybe_builtin_cmd(s):
    if NEXT_PAT.search(s): return "NEXT"
    if PREV_PAT.search(s): return "PREV"
    m = GOTO_PAT.search(s)
    return f"GOTO:{int(m.group(1))}" if m else None

def emit_cmd_and_print(tag):
    global CURRENT_SLIDE
    if tag == "NEXT":
        print("LLM: NEXT\nAdvancing to next slide.\n", flush=True)
        send_slide_cmd("NEXT")
        CURRENT_SLIDE = min(CURRENT_SLIDE + 1, len(SLIDES))
    elif tag == "PREV":
        print("LLM: PREV\nGoing back one slide.\n", flush=True)
        send_slide_cmd("PREV")
        CURRENT_SLIDE = max(CURRENT_SLIDE - 1, 1)
    elif tag.startswith("GOTO:"):
        try:
            idx = int(tag.split(":", 1)[1])
        except:
            idx = None
        if idx:
            print(f"LLM: GOTO:{idx}\nJumping to slide {idx}: {SLIDES[idx-1]['title']}\n", flush=True)
            goto_slide(idx)

# ---------- Multimodal similarity ----------
def semantic_best():
    text = recent_context(10).strip()
    if not text or not SLIDES:
        return None, 0.0, 0.0

    # Query embeddings
    q_mp = embed_text_mpnet(text)
    q_clip = embed_text_clip(text) if _CLIP_AVAILABLE else None

    sims = []
    for i in range(len(SLIDES)):
        s = 0.0
        # text→text (MPNet)
        if TEXT_SIM_WEIGHT > 0 and i < len(EMB["mpnet_text"]):
            s += TEXT_SIM_WEIGHT * cosine(q_mp, EMB["mpnet_text"][i])

        # CLIP (text→image and/or text→text-in-CLIP)
        if CLIP_SIM_WEIGHT > 0 and _CLIP_AVAILABLE and q_clip is not None:
            si = 0.0
            # text→image
            vi = EMB["clip_image"][i]
            if vi.ndim > 1 or vi.size > 1:
                si = max(si, cosine(q_clip, vi))
            # text→text (CLIP space)
            vt = EMB["clip_text"][i]
            if vt.ndim > 1 or vt.size > 1:
                si = max(si, cosine(q_clip, vt))
            s += CLIP_SIM_WEIGHT * si

        sims.append(s)

    order = np.argsort(sims)[::-1]
    if order.size == 0:
        return None, 0.0, 0.0
    best = int(order[0]) + 1
    best_sim = float(sims[order[0]])
    second = float(sims[order[1]]) if len(order) > 1 else 0.0
    margin = best_sim - second
    return best, best_sim, margin

def semantic_route_if_ready():
    global _last_sem_check, _best_idx_stable, _best_idx_since
    ts = now()
    if ts - _last_sem_check < SEM_MIN_GAP_SEC:
        return
    _last_sem_check = ts
    best, sim, margin = semantic_best()

    print(f"[sem] best={best} sim={sim:.2f} margin={margin:.2f} current={CURRENT_SLIDE}", flush=True)

    if not best or best == CURRENT_SLIDE:
        _best_idx_stable = None
        return

    if _best_idx_stable != best:
        _best_idx_stable = best
        _best_idx_since = ts
        return

    if ts - _best_idx_since >= SEM_STEADY_SEC:
        print(f"LLM: GOTO:{best}\nSemantic route (sim={sim:.2f}) → {SLIDES[best-1]['title']}\n", flush=True)
        goto_slide(best)
        _best_idx_stable = None

# ---------- Main loop ----------
def handle_line(raw):
    global last_call, last_print
    ts = now()
    line = raw.strip()
    if not line:
        return
    print(f"ASR: {line}", flush=True)
    if line.lower() == last_print.lower():
        return
    last_print = line
    buf.append((ts, line))
    prune()

    cmd = maybe_builtin_cmd(line)
    if cmd:
        emit_cmd_and_print(cmd)
        last_call = ts
        return
    semantic_route_if_ready()

def main():
    print("LLM consumer ready. Reading transcript from stdin…", file=sys.stderr)
    print(f"[slides] Loaded {len(SLIDES)} slides from {HTML_DECK_PATH}", file=sys.stderr)
    if not _CLIP_AVAILABLE:
        print("[warn] CLIP unavailable — visual routing disabled (install sentence-transformers).", file=sys.stderr)
    for raw in sys.stdin:
        handle_line(raw)

if __name__ == "__main__":
    main()