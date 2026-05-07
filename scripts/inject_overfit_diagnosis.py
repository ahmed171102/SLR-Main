"""
Inject a new Section 4.5 into SLR_Diagnostics.ipynb:
  - Train accuracy vs Test accuracy (gap = overfitting signal)
  - Learning curve from scratch (subsample-based, no saved history needed)
  - Generalisation gap classification: Underfit / Good / Overfit / Severe Overfit
  - Bias-Variance breakdown per class
  - Actionable next-step recommendation box
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

DIAG_PATH = (
    r"C:\Users\HADEEL GAMALELDIN\Desktop\SLR-Main"
    r"\Letters_ORIGINAL\Base_Pipeline_English_Letters\SLR_Diagnostics.ipynb"
)

with open(DIAG_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Helper ─────────────────────────────────────────────────────────────────────
def make_md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def make_code_cell(src):
    lines = src.splitlines(keepends=True)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1].rstrip('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

# ──────────────────────────────────────────────────────────────────────────────
# MARKDOWN HEADER
# ──────────────────────────────────────────────────────────────────────────────
MD_HEADER = """\
## 4.5  Over / Under-Fitting Diagnosis

**Goal:** detect whether poor live accuracy comes from
*underfitting* (model not learning at all) or
*overfitting* (model memorised training data, fails on new data).

| Signal | Likely Cause |
|---|---|
| Train acc ≈ Test acc, both LOW | **Underfitting** — model too simple or features wrong |
| Train acc >> Test acc | **Overfitting** — model memorised training set |
| Train acc ≈ Test acc, both HIGH | **Good generalisation** (live issue = feature mismatch, not model) |
| Gap > 10 % | Overfitting, needs regularisation or more data |

Three sub-analyses:
1. **Train vs Test accuracy** (gap = overfitting signal)
2. **Learning curve** (accuracy as a function of training-set size)
3. **Per-class bias / variance** (which classes suffer most)
"""

# ──────────────────────────────────────────────────────────────────────────────
# CODE CELL
# ──────────────────────────────────────────────────────────────────────────────
CODE_DIAG = '''\
# ─────────────────────────────────────────────────────────────────────────────
# Section 4.5 — Over / Under-fitting Diagnosis
# Requires: model, X_train, X_test, y_train, y_test, CLASS_LABELS
# (all defined in earlier cells)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score

# ── 1. TRAIN vs TEST ACCURACY ─────────────────────────────────────────────────
print("="*60)
print("  OVER / UNDER-FITTING DIAGNOSIS")
print("="*60)

train_probs = model.predict(X_train, batch_size=512, verbose=0)
train_pred  = np.argmax(train_probs, axis=1)
train_true  = np.argmax(y_train, axis=1) if y_train.ndim == 2 else y_train

test_probs  = model.predict(X_test,  batch_size=512, verbose=0)
test_pred   = np.argmax(test_probs,  axis=1)
test_true   = np.argmax(y_test,  axis=1) if y_test.ndim == 2 else y_test

train_acc = accuracy_score(train_true, train_pred)
test_acc  = accuracy_score(test_true,  test_pred)
gap       = train_acc - test_acc

print(f"  Train accuracy : {train_acc*100:.2f}%")
print(f"  Test  accuracy : {test_acc*100:.2f}%")
print(f"  Gap            : {gap*100:.2f}%")
print()

# ── Verdict ───────────────────────────────────────────────────────────────────
if train_acc < 0.80 and test_acc < 0.80:
    verdict = "UNDERFITTING"
    verdict_color = "#e74c3c"
    advice = (
        "Both train and test accuracy are LOW.\\n"
        "→ Model is too simple, OR features are wrong/misaligned.\\n"
        "  Check: Is extract_features() normalizing the same way as training?"
    )
elif gap > 0.15:
    verdict = "SEVERE OVERFITTING"
    verdict_color = "#c0392b"
    advice = (
        f"Train is {gap*100:.1f}% higher than test — model memorised training data.\\n"
        "→ Increase Dropout (try 0.4–0.5), add L2 weight decay, or get more data.\\n"
        "→ Alternatively, reduce model capacity (fewer neurons)."
    )
elif gap > 0.07:
    verdict = "MODERATE OVERFITTING"
    verdict_color = "#e67e22"
    advice = (
        f"Train is {gap*100:.1f}% higher than test — some overfitting present.\\n"
        "→ Increase Dropout slightly (current: 0.2 → try 0.3).\\n"
        "→ Add more training samples for low-recall classes (see Cell 11)."
    )
elif test_acc > 0.88:
    verdict = "GOOD GENERALISATION"
    verdict_color = "#27ae60"
    advice = (
        "Model generalises well on the held-out test set.\\n"
        "→ If live accuracy is still low, the problem is FEATURE DISTRIBUTION SHIFT:\\n"
        "  Live camera landmarks are in screen-space; training was on cropped images.\\n"
        "  Fix: use wrist-relative + scale normalization in extract_features()."
    )
else:
    verdict = "MILD OVERFITTING / ROOM TO IMPROVE"
    verdict_color = "#f39c12"
    advice = (
        f"Gap of {gap*100:.1f}% is acceptable, but test accuracy ({test_acc*100:.1f}%) "
        "could be higher.\\n"
        "→ Gather ~500 more samples per weak class (see per-class chart above).\\n"
        "→ Try data augmentation: slight landmark jitter during training."
    )

print(f"  VERDICT: {verdict}")
print()
print("  ADVICE:")
for line in advice.split("\\n"):
    print("  ", line)
print()

# ── 2. LEARNING CURVE ────────────────────────────────────────────────────────
# Evaluate at 10%, 20%, ..., 100% of training data (fast subsample, no retraining)
# This shows whether MORE DATA would help (underfitting: yes; overfitting: yes too)
fractions = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
lc_train, lc_test = [], []
rng = np.random.default_rng(42)

print("  Computing learning curve (subsampled, no retraining)...")
for frac in fractions:
    n = max(int(len(X_train) * frac), 1)
    idx = rng.choice(len(X_train), n, replace=False)
    p_tr = model.predict(X_train[idx], batch_size=512, verbose=0)
    y_tr = train_true[idx]
    lc_train.append(accuracy_score(y_tr, np.argmax(p_tr, axis=1)))
    lc_test.append(test_acc)   # test acc stays fixed

lc_sizes = [int(f * len(X_train)) for f in fractions]

# ── 3. PER-CLASS BIAS/VARIANCE ───────────────────────────────────────────────
# Per class: train recall vs test recall → gap tells you which classes overfit
from sklearn.metrics import classification_report
train_report = classification_report(train_true, train_pred,
                                     target_names=CLASS_LABELS, output_dict=True)
test_report  = classification_report(test_true,  test_pred,
                                     target_names=CLASS_LABELS, output_dict=True)

class_train_recall = [train_report[c]["recall"] for c in CLASS_LABELS]
class_test_recall  = [test_report[c]["recall"]  for c in CLASS_LABELS]
class_gap          = [tr - te for tr, te in zip(class_train_recall, class_test_recall)]

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle(
    f"Over/Under-fitting Diagnosis  —  Verdict: {verdict}",
    fontsize=15, fontweight="bold", color=verdict_color, y=1.01
)

# ── Panel A: Train vs Test bar ─────────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
bars = ax1.bar(["Train", "Test"], [train_acc, test_acc],
               color=[verdict_color if gap > 0.07 else "#27ae60", "#2980b9"],
               width=0.4, edgecolor="white", linewidth=1.5)
ax1.set_ylim(0, 1.05)
ax1.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% target")
for bar, val in zip(bars, [train_acc, test_acc]):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val*100:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax1.set_title("Train vs Test Accuracy", fontsize=12)
ax1.set_ylabel("Accuracy")
ax1.legend(fontsize=9)
ax1.spines[["top","right"]].set_visible(False)

# Verdict box
verdict_box = (
    f"Verdict: {verdict}\\n"
    f"Gap: {gap*100:.1f}%"
)
ax1.text(0.5, 0.15, verdict_box, transform=ax1.transAxes,
         ha="center", va="bottom", fontsize=10, fontweight="bold",
         color=verdict_color,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor=verdict_color, linewidth=2))

# ── Panel B: Learning curve ────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(lc_sizes, [v*100 for v in lc_train], "o-", color="#e74c3c",
         linewidth=2, markersize=6, label="Train acc (subsampled)")
ax2.axhline(test_acc*100, color="#2980b9", linestyle="--",
            linewidth=2, label=f"Test acc ({test_acc*100:.1f}%)")
ax2.fill_between(lc_sizes,
                 [v*100 for v in lc_train],
                 [test_acc*100]*len(lc_sizes),
                 alpha=0.12, color="#e74c3c")
ax2.set_xlabel("Training samples used")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Learning Curve\\n(does more data help?)", fontsize=12)
ax2.legend(fontsize=9)
ax2.spines[["top","right"]].set_visible(False)

# Annotation: if train curve is still rising → more data would help
slope = lc_train[-1] - lc_train[-3]
note = "↑ Still rising → more data would help" if slope > 0.01 else "Plateau → data alone won\\'t fix it"
ax2.annotate(note, xy=(lc_sizes[-1], lc_train[-1]*100),
             xytext=(-80, -30), textcoords="offset points",
             fontsize=8, color="#7f8c8d",
             arrowprops=dict(arrowstyle="->", color="#7f8c8d"))

# ── Panel C: Gap distribution histogram ────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
gap_pct = [g * 100 for g in class_gap]
colors_gap = ["#e74c3c" if g > 10 else "#f39c12" if g > 5 else "#27ae60"
              for g in gap_pct]
ax3.barh(CLASS_LABELS, gap_pct, color=colors_gap, edgecolor="white")
ax3.axvline(0,  color="black",   linewidth=1)
ax3.axvline(10, color="#e74c3c", linewidth=1, linestyle="--", label="10% overfit threshold")
ax3.axvline(5,  color="#f39c12", linewidth=1, linestyle=":",  label="5% mild overfit")
ax3.set_xlabel("Train Recall − Test Recall (%)")
ax3.set_title("Per-Class Overfitting Gap\\n(red > 10% = problem classes)", fontsize=12)
ax3.legend(fontsize=8)
ax3.spines[["top","right"]].set_visible(False)
patch_ok  = mpatches.Patch(color="#27ae60", label="OK (gap < 5%)")
patch_mid = mpatches.Patch(color="#f39c12", label="Mild (5–10%)")
patch_bad = mpatches.Patch(color="#e74c3c", label="Overfit (> 10%)")
ax3.legend(handles=[patch_ok, patch_mid, patch_bad], fontsize=8, loc="lower right")

# ── Panel D: Per-class train vs test recall side by side ──────────────────────
ax4 = fig.add_subplot(2, 1, 2)
x = np.arange(len(CLASS_LABELS))
w = 0.38
ax4.bar(x - w/2, [v*100 for v in class_train_recall], width=w,
        label="Train Recall", color="#e74c3c", alpha=0.85)
ax4.bar(x + w/2, [v*100 for v in class_test_recall],  width=w,
        label="Test Recall",  color="#2980b9", alpha=0.85)
ax4.axhline(80, color="gray", linestyle="--", linewidth=1)
ax4.set_xticks(x)
ax4.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=9)
ax4.set_ylabel("Recall (%)")
ax4.set_title(
    "Per-Class Train vs Test Recall\\n"
    "Large red–blue gap on a class = that class is overfitting",
    fontsize=12
)
ax4.legend(fontsize=10)
ax4.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.show()

# ── Printed summary ────────────────────────────────────────────────────────────
print("="*60)
print("  DETAILED DIAGNOSIS")
print("="*60)
print(f"  Train accuracy  : {train_acc*100:.2f}%")
print(f"  Test  accuracy  : {test_acc*100:.2f}%")
print(f"  Generalisation gap: {gap*100:.2f}%")
print()
overfit_classes = sorted(
    [(CLASS_LABELS[i], class_gap[i]*100) for i in range(len(CLASS_LABELS))
     if class_gap[i] > 0.10],
    key=lambda x: -x[1]
)
if overfit_classes:
    print("  Most overfitting classes (train recall >> test recall):")
    for cls, g in overfit_classes[:8]:
        print(f"    {cls:12s}  gap = {g:.1f}%")
else:
    print("  No severe per-class overfitting detected.")
print()
print("  RECOMMENDED NEXT STEPS:")
for i, line in enumerate(advice.split("\\n"), 1):
    if line.strip():
        print(f"    {i}. {line.strip()}")
print("="*60)
'''

# ──────────────────────────────────────────────────────────────────────────────
# FIND INSERTION POINT — after Cell 10 (full evaluation), before Cell 11 (per-class F1)
# ──────────────────────────────────────────────────────────────────────────────
def find_cell_idx(nb, keyword):
    for i, c in enumerate(nb['cells']):
        if keyword in ''.join(c['source']):
            return i
    return None

# Cell 9 is markdown "## 4. Run Full Evaluation", Cell 10 is the code.
# We insert AFTER cell 10 (the evaluation code) and BEFORE cell 11 (markdown "## 5.")
insert_after = find_cell_idx(nb, 'y_pred    = np.argmax')  # Cell 10
assert insert_after is not None, "Could not find evaluation cell"

print(f"Inserting after cell {insert_after}")

new_cells = [make_md_cell(MD_HEADER), make_code_cell(CODE_DIAG)]
nb['cells'] = (
    nb['cells'][:insert_after + 1]
    + new_cells
    + nb['cells'][insert_after + 1:]
)

with open(DIAG_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("SUCCESS — 2 new cells inserted into SLR_Diagnostics.ipynb")
print("  Section 4.5: Over/Under-Fitting Diagnosis")
print()
print("Cells added:")
print("  [markdown]  Section 4.5 header + interpretation table")
print("  [code]      Plots: Train vs Test bar, Learning Curve,")
print("              Per-class gap histogram, Train vs Test recall chart")
print("              + printed verdict + actionable advice")
print()
print("Run: Kernel -> Restart & Run All")
