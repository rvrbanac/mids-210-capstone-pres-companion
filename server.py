# server.py
import os, time, subprocess
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

SEND_RIGHT_PATH = os.getenv("SEND_RIGHT_PATH", "./send_right")
SEND_LEFT_PATH  = os.getenv("SEND_LEFT_PATH",  "./send_left")   # optional
DEBOUNCE_MS = 500
_last = {"next": 0.0, "prev": 0.0}

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run(cmd, label):
    log(f"Executing {label}: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)

def run_swift_helper(path, label):
    if not os.path.exists(path):
        return False, f"{label} helper not found at {path}"
    res = run([path], label)
    if res.returncode != 0:
        return False, f"{label} exit {res.returncode}; stdout={res.stdout!r} stderr={res.stderr!r}"
    return True, f"{label} ok; stdout={res.stdout!r}"

def run_applescript_next():
    # Try native PPT slideshow command; fall back to System Events right-arrow
    script = r'''
    tell application "Microsoft PowerPoint"
        try
            if (count of slide show windows) > 0 then
                set w to slide show window 1
                tell w to go to next slide slide show view
            else
                error "no slideshow window"
            end if
        on error
            tell application "System Events" to key code 124 -- Right Arrow
        end try
    end tell
    '''
    res = run(["osascript", "-e", script], "AppleScript NEXT")
    return res.returncode == 0, (res.stdout or res.stderr or "").strip()

def run_applescript_prev():
    script = r'''
    tell application "Microsoft PowerPoint"
        try
            if (count of slide show windows) > 0 then
                set w to slide show window 1
                tell w to go to previous slide slide show view
            else
                error "no slideshow window"
            end if
        on error
            tell application "System Events" to key code 123 -- Left Arrow
        end try
    end tell
    '''
    res = run(["osascript", "-e", script], "AppleScript PREV")
    return res.returncode == 0, (res.stdout or res.stderr or "").strip()

def debounced(kind: str) -> bool:
    now = time.time()
    if (now - _last.get(kind, 0.0)) * 1000 < DEBOUNCE_MS:
        log(f"Debounced {kind}")
        return True
    _last[kind] = now
    return False

@app.post("/command")
async def command(req: Request):
    payload = await req.json()
    cmd = (payload.get("cmd") or "").lower()
    log(f"Incoming command: {payload}")

    if cmd == "next":
        if debounced("next"):
            return JSONResponse({"ok": True, "detail": "debounced"}, status_code=200)

        ok1, d1 = run_swift_helper(SEND_RIGHT_PATH, "NEXT(CGEvent)")
        log(d1)
        ok2, d2 = run_applescript_next()
        log(f"AppleScript NEXT -> {ok2}: {d2}")
        return {"ok": ok1 or ok2, "swift": ok1, "as": ok2}

    if cmd == "prev":
        if debounced("prev"):
            return JSONResponse({"ok": True, "detail": "debounced"}, status_code=200)

        if os.path.exists(SEND_LEFT_PATH):
            ok1, d1 = run_swift_helper(SEND_LEFT_PATH, "PREV(CGEvent)")
            log(d1)
        else:
            ok1, d1 = False, "send_left not found; using AppleScript only"

        ok2, d2 = run_applescript_prev()
        log(f"AppleScript PREV -> {ok2}: {d2}")
        return {"ok": ok1 or ok2, "swift": ok1, "as": ok2, "swift_detail": d1}

    if cmd == "goto":
        idx = payload.get("index")
        log(f"GOTO requested {idx} (not implemented for slideshow; use Office.js in edit/reading view)")
        return {"ok": False, "detail": "goto not supported in full Slide Show"}

    return JSONResponse({"ok": False, "detail": "unknown command"}, status_code=400)

if __name__ == "__main__":
    log(f"Using SEND_RIGHT_PATH={os.path.abspath(SEND_RIGHT_PATH)}")
    if os.path.exists(SEND_LEFT_PATH):
        log(f"Using SEND_LEFT_PATH={os.path.abspath(SEND_LEFT_PATH)}")
    uvicorn.run(app, host="127.0.0.1", port=8765)
