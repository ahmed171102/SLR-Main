import json
import h5py

try:
    with h5py.File('asl_word_lstm_model_final.h5', 'r') as f:
        config = f.attrs.get('model_config')
        if config is not None:
            c_dict = json.loads(config)
            print("Model config shape details:")
            layers = c_dict.get('config', {}).get('layers', [])
            if layers:
                if 'batch_input_shape' in layers[0].get('config', {}):
                    print(layers[0]['config']['batch_input_shape'])
            else:
                print("No layers config found")
except Exception as e:
    print(f"Error: {e}")
