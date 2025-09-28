import queue
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ---------------- Settings ----------------
MODEL_NAME = "small.en"          # good speed/accuracy for English
COMPUTE_TYPE = "int8"            # int8 = faster on Apple Silicon; try "float16" for max accuracy
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2               # smaller = snappier updates
OVERLAP_SECONDS = 0.5            # how much of the last chunk to keep
VAD_FILTER = True                # skip non-speech
NO_SPEECH_THRESHOLD = 0.45       # raise if you get noise transcripts
SILENCE_ANNOUNCE = 2.0           # print silence when gap >= N seconds
# ------------------------------------------

print("Loading model…", file=sys.stderr, flush=True)
model = WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE)
print(f"Model loaded: {MODEL_NAME}, compute_type={COMPUTE_TYPE}", file=sys.stderr)

audio_q = queue.Queue()
buffer = np.zeros(0, dtype=np.float32)
running_context = ""  # keep text history to stabilize terms

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr, flush=True)
    mono = indata.mean(axis=1).astype(np.float32)  # convert to mono float32
    audio_q.put(mono)

def main():
    global buffer, running_context
    chunk_samples = SAMPLE_RATE * CHUNK_SECONDS

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        print("🎙️  Listening… Ctrl+C to stop.", file=sys.stderr)
        while True:
            # Collect audio until we have one chunk
            while buffer.size < chunk_samples:
                buffer = np.concatenate([buffer, audio_q.get()])

            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]

            # Transcribe this chunk
            segments, _ = model.transcribe(
                chunk,
                language="en",
                vad_filter=VAD_FILTER,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                condition_on_previous_text=True,
                initial_prompt=running_context[-300:]  # last 300 chars context
            )

            # Process results
            last_end = None
            out = []
            for seg in segments:
                if last_end is not None:
                    gap = seg.start - last_end
                    if gap >= SILENCE_ANNOUNCE:
                        out.append(f"[silence {gap:.1f}s]")
                if seg.text.strip():
                    out.append(seg.text.strip())
                last_end = seg.end

            if out:
                line = " ".join(out)
                running_context += (" " if running_context else "") + line
                print(line, flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
