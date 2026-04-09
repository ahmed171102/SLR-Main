import json
import sys

def extract(in_file, out_file):
    with open(in_file, encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(out_file, 'w', encoding='utf-8') as out:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                # source can be a string or a list of strings
                src = cell['source']
                if isinstance(src, list):
                    src = "".join(src)
                out.write(src)
                out.write('\n\n')

if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2])
