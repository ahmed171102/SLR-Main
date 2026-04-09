import json, os, glob

root = r'M:\Term 10\Grad'
results = []

for f in glob.glob(os.path.join(root, '**', '*.ipynb'), recursive=True):
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
            nb = json.load(fp)
        cells = nb.get('cells', [])
        ncells = len(cells)
        
        # Get first 3 code/markdown cells for purpose detection
        first_sources = []
        for c in cells[:5]:
            src = ''.join(c.get('source', []))
            first_sources.append(src[:300])
        
        combined = '\n'.join(first_sources).lower()
        
        # Detect purpose
        purpose = 'unknown'
        if 'live_test' in f.lower() or 'live test' in f.lower() or 'camera' in combined or 'cap = cv2' in combined or 'videocapture' in combined:
            purpose = 'live-test/inference'
        elif 'download' in f.lower() or 'download' in combined:
            purpose = 'dataset-download'
        elif 'fixer' in f.lower() or 'fix' in combined[:200]:
            purpose = 'data-fixing/cleaning'
        elif 'merge' in f.lower() or 'vocab' in f.lower():
            purpose = 'data-merging/vocab'
        elif 'combined_architecture' in f.lower() or 'core_arch' in f.lower():
            purpose = 'combined-architecture'
        elif 'draft' in f.lower():
            purpose = 'draft/exploration'
        elif 'trial' in f.lower():
            purpose = 'experimental-trial'
        elif 'wlasl' in f.lower():
            purpose = 'dataset-analysis'
        elif 'training' in f.lower() or 'model.fit' in combined or 'model.compile' in combined:
            purpose = 'model-training'
        elif 'cbn' in f.lower():
            purpose = 'model-training'
        
        # Detect architecture
        arch = 'none'
        full_text = ''
        for c in cells:
            full_text += ''.join(c.get('source', [])).lower()
        
        if 'mobilenetv2' in full_text or 'mobilenet' in full_text:
            arch = 'MobileNetV2'
        if 'lstm' in full_text:
            if arch != 'none':
                arch += '+LSTM'
            else:
                arch = 'LSTM'
        if 'transformer' in full_text or 'multiheadattention' in full_text:
            if arch != 'none':
                arch += '+Transformer'
            else:
                arch = 'Transformer'
        if 'mediapipe' in full_text and 'mlp' in full_text:
            if arch == 'none':
                arch = 'MLP (MediaPipe)'
        if 'mediapipe' in full_text and arch == 'none':
            if 'dense' in full_text or 'sequential' in full_text:
                arch = 'MLP (MediaPipe)'
        if 'cbn' in f.lower():
            arch = 'CNN-based'
        
        # Detect dataset
        dataset = 'unknown'
        if 'wlasl' in full_text:
            dataset = 'WLASL'
        if 'arsl' in full_text or 'arabic' in f.lower() or 'arabic' in full_text:
            if 'word' in f.lower():
                dataset = 'ArSL-Word (KArSL/custom)'
            else:
                dataset = 'ArSL-Letter (Arabic Sign Language)'
        if 'asl' in f.lower() and 'arsl' not in f.lower():
            if 'word' in f.lower():
                dataset = 'WLASL/ASL-Word'
            elif 'letter' in f.lower() or 'asl_alphabet' in full_text or 'mediapipe_keypoints' in full_text:
                dataset = 'ASL-Letter (Kaggle alphabet)'
        if 'asl_alphabet' in full_text:
            dataset = 'ASL-Letter (Kaggle alphabet)'
        if 'kaggle' in f.lower():
            if 'word' in f.lower() and 'arsl' in f.lower():
                dataset = 'ArSL-Word (KArSL Kaggle)'
            elif 'word' in f.lower():
                dataset = 'WLASL/ASL-Word (Kaggle)'
        
        rel = os.path.relpath(f, root)
        size_kb = os.path.getsize(f) / 1024
        results.append(f"{rel}|{size_kb:.1f}|{ncells}|{purpose}|{arch}|{dataset}")
    except Exception as e:
        rel = os.path.relpath(f, root)
        results.append(f"{rel}|?|?|ERROR: {str(e)[:50]}|?|?")

# Also add wlasl-data.ipynb if at root
wlasl = os.path.join(root, 'wlasl-data.ipynb')
if os.path.exists(wlasl) and not any('wlasl-data.ipynb' in r for r in results):
    try:
        with open(wlasl, 'r', encoding='utf-8', errors='ignore') as fp:
            nb = json.load(fp)
        cells = nb.get('cells', [])
        size_kb = os.path.getsize(wlasl) / 1024
        results.append(f"wlasl-data.ipynb|{size_kb:.1f}|{len(cells)}|dataset-analysis|none|WLASL")
    except:
        pass

with open(os.path.join(root, '_audit_results.txt'), 'w', encoding='utf-8') as f:
    f.write('path|size_kb|cells|purpose|architecture|dataset\n')
    for r in results:
        f.write(r + '\n')

print(f"Done. {len(results)} notebooks analyzed.")
