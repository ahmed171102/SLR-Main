import numpy as np

try:
    data = np.load('asl_word_sequences.npz')
    print("X Shape:", data['X'].shape)
    with open('output_shape.txt', 'w') as f:
        f.write(str(data['X'].shape))
except Exception as e:
    print(f"Error: {e}")
