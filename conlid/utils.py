import json
import pickle

from pathlib import Path

def load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_json(path: Path):
    with open(path, "r") as json_file:
        return json.load(json_file)

def save_pickle(vec, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path,  'wb+') as f:
        _ = pickle.dump(vec, f)

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def save_txt(vec, path: Path, new_line=True):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as f:
        for item in vec:
            if new_line:
                f.write(str(item) + '\n')
            else:
                f.write(str(item))