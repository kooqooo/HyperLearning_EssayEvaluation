import os
import json

import pandas as pd
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "024.에세이 글 평가 데이터")
DATA_DIR = os.path.join(DATA_DIR, "01.데이터", "1.Training", "라벨링데이터")

length = []
for subdir in os.listdir(DATA_DIR):
    subdir_path = os.path.join(DATA_DIR, subdir)
    if not os.path.isdir(subdir_path):
        continue
    for file in tqdm(os.listdir(subdir_path)):
        if not file.endswith('.json'):
            continue
        json_path = os.path.join(subdir_path, file)
        csv_path = os.path.join(subdir_path, file.replace('.json', '.csv'))
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        length.append(data["info"]["essay_len"])

print(len(length))