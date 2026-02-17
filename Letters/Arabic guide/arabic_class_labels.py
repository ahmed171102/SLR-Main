# Arabic Sign Language Class Labels

# Arabic Alphabet (28 letters)
ARABIC_LETTERS = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر',
    'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
    'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'
]

# Additional Arabic characters
ARABIC_SPECIAL = ['ء', 'ة', 'ى']  # Hamza, Taa Marbuta, Alef Maqsura

# Arabic numbers (0-9)
ARABIC_NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Common words in Arabic Sign Language (optional - add later)
ARABIC_WORDS = [
    'مرحبا',    # Hello
    'شكراً',     # Thanks
    'نعم',      # Yes
    'لا',       # No
    'من فضلك'   # Please
]

# Control characters (same as English version)
ARABIC_CONTROL = ['space', 'del', 'nothing']

# Complete class list for basic implementation (letters + control)
ARABIC_CLASSES = ARABIC_LETTERS + ARABIC_CONTROL

# Extended class list (if you want to include numbers and special characters)
ARABIC_CLASSES_EXTENDED = ARABIC_LETTERS + ARABIC_SPECIAL + ARABIC_NUMBERS + ARABIC_CONTROL

# Class labels dictionary for easy lookup
ARABIC_CLASS_DICT = {i: letter for i, letter in enumerate(ARABIC_CLASSES)}

print(f"Arabic Sign Language Classes:")
print(f"  - Letters: {len(ARABIC_LETTERS)}")
print(f"  - Total classes: {len(ARABIC_CLASSES)}")
print(f"  - Classes: {ARABIC_CLASSES}")







