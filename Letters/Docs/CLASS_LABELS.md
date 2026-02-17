# Complete Letter Class Labels — English & Arabic
# قائمة الحروف الكاملة — إنجليزي وعربي

---

## English ASL Letters (29 Classes)

| # | Label | Type | Description |
|---|-------|------|-------------|
| 1 | A | Letter | |
| 2 | B | Letter | |
| 3 | C | Letter | |
| 4 | D | Letter | |
| 5 | E | Letter | |
| 6 | F | Letter | |
| 7 | G | Letter | |
| 8 | H | Letter | |
| 9 | I | Letter | |
| 10 | J | Letter | Involves motion |
| 11 | K | Letter | |
| 12 | L | Letter | |
| 13 | M | Letter | |
| 14 | N | Letter | |
| 15 | O | Letter | |
| 16 | P | Letter | |
| 17 | Q | Letter | |
| 18 | R | Letter | |
| 19 | S | Letter | |
| 20 | T | Letter | |
| 21 | U | Letter | |
| 22 | V | Letter | |
| 23 | W | Letter | |
| 24 | X | Letter | |
| 25 | Y | Letter | |
| 26 | Z | Letter | Involves motion |
| 27 | del | Control | Delete last character |
| 28 | nothing | Control | No sign / idle |
| 29 | space | Control | Insert space |

> **Note:** J and Z involve hand motion in real ASL. The MLP model uses a single frame, so these may have lower accuracy than static letters.

---

## Arabic ArSL Letters — Core Set (31 Classes)

| # | Arabic | Romanized | Type |
|---|--------|-----------|------|
| 1 | ا | Alef | Letter |
| 2 | ب | Beh | Letter |
| 3 | ت | Teh | Letter |
| 4 | ث | Theh | Letter |
| 5 | ج | Jeem | Letter |
| 6 | ح | Hah | Letter |
| 7 | خ | Khah | Letter |
| 8 | د | Dal | Letter |
| 9 | ذ | Thal | Letter |
| 10 | ر | Reh | Letter |
| 11 | ز | Zain | Letter |
| 12 | س | Seen | Letter |
| 13 | ش | Sheen | Letter |
| 14 | ص | Sad | Letter |
| 15 | ض | Dad | Letter |
| 16 | ط | Tah | Letter |
| 17 | ظ | Zah | Letter |
| 18 | ع | Ain | Letter |
| 19 | غ | Ghain | Letter |
| 20 | ف | Feh | Letter |
| 21 | ق | Qaf | Letter |
| 22 | ك | Kaf | Letter |
| 23 | ل | Lam | Letter |
| 24 | م | Meem | Letter |
| 25 | ن | Noon | Letter |
| 26 | ه | Heh | Letter |
| 27 | و | Waw | Letter |
| 28 | ي | Yeh | Letter |
| 29 | space | — | Control |
| 30 | del | — | Control |
| 31 | nothing | — | Control |

### Extended Arabic Set (34 Classes — some models)

Some training CSVs include additional labels:

| # | Label | Arabic | Notes |
|---|-------|--------|-------|
| 32 | Al | ال | Alef + Lam (definite article) |
| 33 | Laa | لا | Lam + Alef ligature |
| 34 | Teh_Marbuta | ة | Taa Marbuta (feminine ending) |

> **Note:** The Arabic class count varies between notebooks (31, 34, or 35) due to different CSV versions. The `arabic_class_labels.py` file defines the canonical 31-class set.

---

## Special Arabic Characters (defined in `arabic_class_labels.py`)

```python
ARABIC_SPECIAL = ['ء', 'ة', 'ى']  # Hamza, Taa Marbuta, Alef Maqsura
ARABIC_NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ARABIC_WORDS = ['مرحبا', 'شكراً', 'نعم', 'لا', 'من فضلك']
```

These are defined for future expansion but **not currently used** in training.

---

## Label Display Mapping (Arabic Combined Notebook)

The Arabic Combined notebook uses romanized labels internally and maps them to Arabic for display:

```python
NAME_TO_ARABIC = {
    'Alef': 'ا', 'Beh': 'ب', 'Teh': 'ت', 'Theh': 'ث',
    'Jeem': 'ج', 'Hah': 'ح', 'Khah': 'خ', 'Dal': 'د',
    'Thal': 'ذ', 'Reh': 'ر', 'Zain': 'ز', 'Seen': 'س',
    'Sheen': 'ش', 'Sad': 'ص', 'Dad': 'ض', 'Tah': 'ط',
    'Zah': 'ظ', 'Ain': 'ع', 'Ghain': 'غ', 'Feh': 'ف',
    'Qaf': 'ق', 'Kaf': 'ك', 'Lam': 'ل', 'Meem': 'م',
    'Noon': 'ن', 'Heh': 'ه', 'Waw': 'و', 'Yeh': 'ي',
    'space': ' ', 'del': 'del', 'nothing': None
}
```

---

## Summary

| Language | Core Classes | With Extensions | Model Files |
|---|---|---|---|
| English (ASL) | 29 | 29 | `asl_mediapipe_mlp_model.h5` |
| Arabic (ArSL) | 31 | 34 | `arsl_mediapipe_mlp_model_best.h5` |
