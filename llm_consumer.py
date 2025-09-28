#!/usr/bin/env python3
"""
llm_consumer.py
Reads live captions from stdin, always prints them (prefixed 'ASR:'),
detects slide commands like "next slide please", and calls a local LLM
(Mistral via llama.cpp through LangChain's LlamaCpp) for questions.
Outputs LLM actions/responses prefixed with 'LLM:'.
"""

import sys
import time
import re
import collections
import requests                              # NEW: to talk to server.py
from langchain_community.llms import LlamaCpp  # llama.cpp backend

# ========= Configure your local model =========
MODEL_PATH = '/Users/rossvrbanac/MIDS/267Summer25/Assignment 5/Mistral-7B-Instruct-v0.3.fp16.gguf'

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_gpu_layers=1000,   # Reduce if you hit VRAM limits
    n_threads=16,
    n_batch=2048,
    temperature=0.2,     # crisp answers
    top_p=0.95,
    max_tokens=256,
    repeat_penalty=1.1,
    verbose=False,
)
# =============================================

# ---- Slide server config ----
SERVER_URL = "http://127.0.0.1:8765/command"  # matches server.py
CMD_DEBOUNCE_S = 0.45                          # avoid rapid double-advance
_last_cmd_sent = (None, 0.0)                   # (("NEXT"/"PREV"/("GOTO",n)), ts)

def send_slide_cmd(kind: str, index: int | None = None):
    """
    kind: "NEXT" | "PREV" | "GOTO"
    index: required for GOTO
    """
    global _last_cmd_sent
    now_ts = time.time()
    last_kind, last_ts = _last_cmd_sent

    # coalesce identical back-to-back commands within debounce window
    same = last_kind == (kind if kind != "GOTO" else ("GOTO", index))
    if same and (now_ts - last_ts) < CMD_DEBOUNCE_S:
        return

    payload = {"cmd": "next" if kind == "NEXT"
               else "prev" if kind == "PREV"
               else "goto"}
    if kind == "GOTO" and isinstance(index, int):
        payload["index"] = index

    try:
        requests.post(SERVER_URL, json=payload, timeout=0.3)
    except Exception as e:
        # Non-fatal: keep presenting even if server is down
        print(f"LLM: (warn) slide server unreachable: {e}", flush=True)

    _last_cmd_sent = ((kind if kind != "GOTO" else ("GOTO", index)), now_ts)

# ---- Heuristics ----
Q_PAT = re.compile(
    r"\?|^(can|could|would|should|what|why|how|when|where|which|who|explain|tell me)\b",
    re.IGNORECASE
)

NEXT_PAT = re.compile(
    r"\b(next\s+slide(?!s)\b|next please|go\s*next|advance\s+the\s+slide|move\s+on)\b",
    re.IGNORECASE
)
PREV_PAT = re.compile(
    r"\b(previous\s+slide|prev\s+slide|go\s*back|back\s+one)\b",
    re.IGNORECASE
)
GOTO_PAT = re.compile(
    r"\bslide\s*(\d{1,3})\b",  # e.g., "slide 12"
    re.IGNORECASE
)

ROLLING_SECONDS = 45
MIN_GAP_S = 0.3
COALESCE_S = 0.7

buf = collections.deque()      # (timestamp, text)
last_call = 0.0
last_print = ""                # dedupe repeated identical lines

def now() -> float:
    return time.time()

def prune():
    cutoff = now() - ROLLING_SECONDS
    while buf and buf[0][0] < cutoff:
        buf.popleft()

def current_context() -> str:
    prune()
    return " ".join(t for _, t in buf)[-3000:]

def looks_like_question(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    return bool(Q_PAT.search(s))

def maybe_builtin_cmd(s: str):
    if NEXT_PAT.search(s):
        return "NEXT"
    if PREV_PAT.search(s):
        return "PREV"
    m = GOTO_PAT.search(s)
    if m:
        return f"GOTO:{int(m.group(1))}"
    return None

def build_prompt(latest: str, ctx: str) -> str:
    return f"""You are a concise assistant helping during a live talk.

Recent transcript (most recent last):
{ctx}

Task:
If the latest utterance is a question or a clear slide command,
answer in 1–3 sentences. If it's a slide command, respond ONLY with one of:
NEXT, PREV, or GOTO:<number>. If there is no clear question/command, respond with NONE.

Latest utterance:
{latest}
"""

def emit_cmd_and_print(tag: str):
    """Helper to print & send slide command."""
    if tag == "NEXT":
        print("LLM: NEXT\nI, the LLM, will turn to the next slide.\n", flush=True)
        send_slide_cmd("NEXT")
    elif tag == "PREV":
        print("LLM: PREV\nI, the LLM, will change the slide.\n", flush=True)
        send_slide_cmd("PREV")
    elif tag.startswith("GOTO:"):
        try:
            idx = int(tag.split(":", 1)[1])
        except Exception:
            idx = None
        print(f"LLM: GOTO:{idx}\nI, the LLM, will change the slide.\n", flush=True)
        if idx is not None:
            send_slide_cmd("GOTO", idx)

def handle_line(raw: str):
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

    # 1) Built-in fast path: no LLM call
    cmd = maybe_builtin_cmd(line)
    if cmd:
        emit_cmd_and_print(cmd)
        last_call = ts
        return

    # 2) Throttle LLM calls
    if ts - last_call < MIN_GAP_S:
        return

    # 3) Ask LLM only when it looks like Q/command
    if looks_like_question(line):
        time.sleep(COALESCE_S)
        ctx = current_context()
        prompt = build_prompt(line, ctx)
        resp = llm.invoke(prompt).strip()

        if resp and resp.upper() != "NONE":
            up = resp.upper().strip()
            if up in ("NEXT", "PREV") or up.startswith("GOTO:"):
                emit_cmd_and_print(up)
            else:
                print(f"LLM: {resp}\n", flush=True)

        last_call = now()

def main():
    print("LLM consumer ready. Reading transcript from stdin…", file=sys.stderr)
    for raw in sys.stdin:
        handle_line(raw)

if __name__ == "__main__":
    main()
