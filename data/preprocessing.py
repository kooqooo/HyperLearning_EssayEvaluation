import os
import json

import pandas as pd
from tqdm import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "024.에세이 글 평가 데이터")
DATA_DIR = os.path.join(DATA_DIR, "01.데이터", "1.Training", "라벨링데이터")

"""
`루브릭_설명서_v0.3.hwp`와 다른 항목이 있어서 데이터에서 값을 직접 불러옵니다.
"""
exp = ["exp_grammar", "exp_vocab", "exp_style"]
org = ["org_essay", "org_paragraph", "org_coherence", "org_quantity"]
con = ["con_clearance", "con_description", "con_novelty", "con_prompt"]


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
            if not file.endswith(".json"):
                continue
            json_path = os.path.join(subdir_path, file)
            
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 문단이 여러개라면 하나의 문단으로 합침
            if len(data["paragraph"]) > 1:
                paragraph = [p["paragraph_txt"] for p in data["paragraph"]]
                paragraph = "".join(paragraph)
            else:
                paragraph = data["paragraph"][0]["paragraph_txt"]

            paragraph = paragraph.replace("#@문장구분#", " ")
            paragraph = " ".join(paragraph.split())
            
            scores = calc_scores(data)

            # 데이터 프레임에 추가할 정보 추출
            essay = {
                "id": data["info"]["essay_id"],
                "type": data["info"]["essay_type"],
                "main_subject": data["info"]["essay_main_subject"],
                "len": data["info"]["essay_len"],
                "level": data["info"]["essay_level"],
                "student_grade": data["student"]["student_grade"],
                "essay": paragraph,
                "exp_score": scores["exp_score"],
                "org_score": scores["org_score"],
                "con_score": scores["con_score"],
            }
            essay_data.append(essay)
    
    return pd.DataFrame(essay_data)

def calc_weighted_score(scores: list, weights: list) -> float:
    return sum(sum(s * w for s, w in zip(score, weights)) for score in scores) / sum(weights) / 3

def calc_scores(data: dict) -> dict[str: float]:
    # print(data)
    rubric = data["rubric"]
    essay_score = data["score"]["essay_scoreT_detail"]
    
    exp_scores = essay_score["essay_scoreT_exp"]
    exp_weights = [rubric["expression_weight"][e] for e in exp]
    exp_score = calc_weighted_score(exp_scores, exp_weights)
    
    org_scores = essay_score["essay_scoreT_org"]
    org_weights = [rubric["organization_weight"][o] for o in org]
    org_score = calc_weighted_score(org_scores, org_weights)

    con_scores = essay_score["essay_scoreT_cont"]
    con_weights = [rubric["content_weight"][c] for c in con]
    con_score = calc_weighted_score(con_scores, con_weights)

    scores = {
        "exp_score": exp_score,
        "org_score": org_score,
        "con_score": con_score,
    }
    return scores

if __name__ == "__main__":
    length = []
    df = process_essay_data(DATA_DIR)
    df.to_csv(os.path.join(ROOT_DIR, "data", "data.csv"), index=False, encoding="utf-8-sig")