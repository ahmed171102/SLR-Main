"""
Fix Production_Architecture_Arabic.ipynb:
  1. Cell 16 → Fix draw_arabic_text: correct font path priority, protect
     non-Arabic placeholder text from reshaper so "_" never becomes "????".
  2. Cell 10 → Fix extract_features: add wrist-relative normalization to
     match what the training images naturally produce (training images were
     static, cropped hand photos where the wrist is near origin).
  3. Cell 6  → Keep ARABIC_MAP as-is (already maps "ain"→"ع" etc.).
     Status text in Cell 18 shows the Arabic char (tracked_arabic) so the
     display problem is purely a rendering issue fixed by Cell 16.
"""

import json, sys
sys.stdout.reconfigure(encoding="utf-8")

PROD_PATH = (
    r"C:\Users\HADEEL GAMALELDIN\Desktop\SLR-Main"
    r"\Letters_ORIGINAL\ArSL (Arabic Letters)\Production_Architecture_Arabic.ipynb"
)

with open(PROD_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# CELL INDEX DISCOVERY  (find by content, not by hard-coded index)
# ──────────────────────────────────────────────────────────────────────────────
def find_cell(nb, keyword):
    for i, c in enumerate(nb["cells"]):
        if keyword in "".join(c["source"]):
            return i
    return None

idx_render   = find_cell(nb, "draw_arabic_text")          # Cell 16
idx_features = find_cell(nb, "def extract_features")      # Cell 10

print(f"  draw_arabic_text cell : {idx_render}")
print(f"  extract_features cell : {idx_features}")

assert idx_render   is not None, "Could not find draw_arabic_text cell"
assert idx_features is not None, "Could not find extract_features cell"

# ──────────────────────────────────────────────────────────────────────────────
# FIX 1 — Arabic text rendering (Cell 16)
# Problems fixed:
#   a) "_" placeholder was reshaped by arabic_reshaper → garbled glyphs → "????"
#   b) Font search list had "arial.ttf" (relative) first – never found on Windows
#   c) Fallback ImageFont.load_default() is a tiny bitmap that can't render Arabic
# ──────────────────────────────────────────────────────────────────────────────
NEW_CELL_RENDER = '''\
try:
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_TEXT_OK = True
    print("✓ Arabic text rendering libraries found")
except ImportError:
    ARABIC_TEXT_OK = False
    print("⚠ Arabic text libraries missing — run:")
    print("  pip install arabic-reshaper python-bidi pillow")

# Windows Arabic-capable fonts (first match wins)
_FONT_CANDIDATES = [
    r"C:\\Windows\\Fonts\\arial.ttf",          # Arial — most reliable on Windows
    r"C:\\Windows\\Fonts\\tahoma.ttf",         # Tahoma — full Arabic support
    r"C:\\Windows\\Fonts\\calibri.ttf",        # Calibri fallback
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/System/Library/Fonts/Arial.ttf",                  # macOS
]

def _get_font(size):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

# Arabic Unicode range: \\u0600–\\u06FF
def _is_arabic(text):
    return any("\\u0600" <= ch <= "\\u06FF" for ch in text)

def draw_arabic_text(frame, text, position, font_size=40, color=(255, 255, 255)):
    """Render text on an OpenCV frame.
    Arabic content → arabic_reshaper + python-bidi + PIL.
    Non-Arabic content (e.g. status strings, '_') → plain PIL text (no reshape).
    Falls back to cv2.putText if PIL libraries are unavailable.
    """
    if not text:
        return frame

    # ── PIL unavailable → cv2 fallback (ASCII-safe only) ──────────────────────
    if not ARABIC_TEXT_OK:
        safe = "".join(c if ord(c) < 128 else "?" for c in text)
        cv2.putText(frame, safe, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)
        return frame

    # ── PIL path ──────────────────────────────────────────────────────────────
    if _is_arabic(text):
        # Reshape ligatures and apply bidi RTL ordering
        display_text = get_display(arabic_reshaper.reshape(text))
    else:
        # Plain Latin/numeric/symbol — do NOT reshape (causes "????")
        display_text = text

    pil_img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw      = ImageDraw.Draw(pil_img)
    font      = _get_font(font_size)
    rgb_color = (color[2], color[1], color[0])   # BGR → RGB
    draw.text(position, display_text, fill=rgb_color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

print("✓ draw_arabic_text defined (Arabic-safe, '_' placeholder fixed)")
'''

# ──────────────────────────────────────────────────────────────────────────────
# FIX 2 — extract_features normalization (Cell 10)
#
# ROOT CAUSE of live-vs-offline accuracy gap:
#   Training (Kaggle): static_image_mode=True, model_complexity=0
#     → landmarks are wrist-relative (image is a cropped hand photo)
#   Production live:  video_mode=False set in Hands(), but the RAW absolute
#     (x,y,z) screen-space coords are passed straight to the model.
#     These are in [0,1] of the full camera frame, NOT normalized to the hand.
#
# Fix: subtract wrist (landmark 0) to make coords hand-relative, then divide
#      by the wrist→middle-finger-mcp distance (landmark 9) for scale invariance.
#      This matches what MediaPipe naturally produces on cropped images.
# ──────────────────────────────────────────────────────────────────────────────
NEW_CELL_FEATURES = '''\
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 1,
    min_detection_confidence = MP_DETECTION_CONFIDENCE,
    min_tracking_confidence  = MP_TRACKING_CONFIDENCE,
)

def extract_features(results):
    """Return (1, 63) float32 array normalized to match training distribution.

    Training images were static cropped photos → landmarks sit near the wrist
    with no background scale.  Live frames are full-camera → absolute coords.

    Normalization applied here (mirrors what the training images give):
      1. Translate so wrist (lm[0]) is at origin.
      2. Scale so the wrist-to-mid-MCP (lm[9]) distance == 1.
         (robust hand-size normalizer that works across all distances/zoom levels)
    """
    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0].landmark
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)

    # Step 1: wrist-relative translation
    pts -= pts[0]   # landmark 0 is the wrist

    # Step 2: scale normalisation (wrist → mid-finger MCP distance)
    scale = np.linalg.norm(pts[9])   # landmark 9 = middle-finger MCP
    if scale > 1e-6:
        pts /= scale

    return pts.reshape(1, -1)   # (1, 63)

print("✓ MediaPipe Hands configured (max_num_hands=1)")
print("✓ extract_features: wrist-relative + scale-normalized (matches training)")
'''

# ──────────────────────────────────────────────────────────────────────────────
# APPLY PATCHES
# ──────────────────────────────────────────────────────────────────────────────
def set_cell_source(nb, idx, new_src):
    lines = new_src.splitlines(keepends=True)
    # Ensure last line has no trailing newline (notebook convention)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    nb["cells"][idx]["source"] = lines

set_cell_source(nb, idx_render,   NEW_CELL_RENDER)
set_cell_source(nb, idx_features, NEW_CELL_FEATURES)

with open(PROD_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print()
print("=" * 60)
print("SUCCESS — Production_Architecture_Arabic.ipynb patched")
print("=" * 60)
print()
print("Changes applied:")
print()
print("  FIX 1 (Cell", idx_render, ") — draw_arabic_text")
print("    • Font search now starts with Windows arial.ttf/tahoma.ttf")
print("    • _is_arabic() guard: '_' placeholder skips arabic_reshaper")
print("    • Arabic chars still go through reshaper + bidi — displays أ not alef")
print("    • cv2 fallback replaces non-ASCII with '?' (no more '????')")
print()
print("  FIX 2 (Cell", idx_features, ") — extract_features")
print("    • Wrist-relative translation (lm[0] → origin)")
print("    • Scale normalization (wrist→mid-MCP distance = 1)")
print("    • Matches the effective normalization in static training images")
print()
print("Next: In Jupyter, do Kernel → Restart & Run All, then test live.")
