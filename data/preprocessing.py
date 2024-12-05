import os
import json

import pandas as pd
from tqdm import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "024.에세이 글 평가 데이터")
DATA_DIR = os.path.join(DATA_DIR, "01.데이터", "1.Training", "라벨링데이터")


def process_essay_data(data_dir: str) -> pd.DataFrame:
    """
    라벨링 데이터 폴더 내의 모든 JSON 파일을 읽어들여 데이터프레임으로 변환

    Args:
        data_dir (str): 라벨링 데이터 폴더 경로

    Returns:
        pd.DataFrame: JSON 파일을 읽어들인 데이터프레임
    """
    essay_data = []
    
    # 라벨링 데이터 폴더 내의 모든 하위 디렉토리에 대해 반복
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # 각 디렉토리에서 모든 JSON 파일을 읽어들임
        for file in tqdm(os.listdir(subdir_path)):
            if not file.endswith('.json'):
                continue
            json_path = os.path.join(subdir_path, file)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터 프레임에 추가할 정보 추출
            essay = {
                'essay_len': data['info']['essay_len'],
            }
            essay_data.append(essay)
    
    return pd.DataFrame(essay_data)

if __name__ == '__main__':
    length = []
    df = process_essay_data(DATA_DIR)
    print(df.head())