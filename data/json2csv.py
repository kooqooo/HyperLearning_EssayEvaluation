import os
import json

import pandas as pd

"""
운영체제별로 다운로드 받은 데이터 경로가 일부 다릅니다.
따라서 '024.에세이'로 시작하는 폴더를 찾아서 데이터 경로를 설정합니다.
Windows: '024.에세이 글 평가 데이터'
macOS: '024.에세이_글_평가_데이터'
"""

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
for folder in os.listdir(ROOT_DIR):
    if folder.startswith("024.에세이"):
        DATA_DIR = os.path.join(ROOT_DIR, folder)
        break
DATA_DIR = os.path.join(DATA_DIR, "01.데이터", "1.Training", "라벨링데이터")

length = []
for subdir in os.listdir(DATA_DIR):
    subdir_path = os.path.join(DATA_DIR, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith('.json'):
                json_path = os.path.join(subdir_path, file)
                csv_path = os.path.join(subdir_path, file.replace('.json', '.csv'))
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                length.append(data["info"]["essay_len"])

print(sorted(length))