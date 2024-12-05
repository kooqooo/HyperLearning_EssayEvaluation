essay_type_to_category = {
    "글짓기" : "수필형",
    "대안제시" : "논술형",
    "설명글" : "수필형",
    "주장글" : "논술형",
    "찬성반대" : "논술형",
}

category_weights = {
    "수필형" : {"org": 3, "con": 3, "exp": 4},
    "논술형" : {"org": 4, "con": 4, "exp": 2},
}

# 논술형 세부 가중치
argumentative_detail_weights_1 = {
    "org": [3, 0, 0, 1],
    "con": [4, 2, 3, 1],
    "exp": [4, 3, 0],
}
argumentative_detail_weights_2 = {
    "org": [7, 0, 2, 1],
    "con": [4, 2, 2, 1],
    "exp": [3, 3, 0],
}
argumentative_detail_weights_3 = {
    "org": [0, 6, 3, 1],
    "con": [3, 3, 3, 1],
    "exp": [2, 4, 4],
}
argumentative_detail_weights_4 = {
    "org": [3, 3, 3, 1],
    "con": [3, 3, 3, 1],
    "exp": [2, 4, 4],
}
argumentative_detail_weights_by_grade = {
    "초등_4학년" : argumentative_detail_weights_1,
    "초등_5학년" : argumentative_detail_weights_1,

    "초등_6학년" : argumentative_detail_weights_2,
    "중등_1학년" : argumentative_detail_weights_2,
    
    "중등_2학년" : argumentative_detail_weights_3,
    "중등_3학년" : argumentative_detail_weights_3,
    "고등_1학년" : argumentative_detail_weights_3,

    "고등_2학년" : argumentative_detail_weights_4,
    "고등_3학년" : argumentative_detail_weights_4,
}

# 수필형 세부 가중치
narrative_detail_weights_1 = {
    "org": [5, 0, 0, 1],
    "con": [4, 2, 0, 4],
    "exp": [4, 3, 0],
}
narrative_detail_weights_2 = {
    "org": [2, 4, 2, 1],
    "con": [4, 2, 0, 4],
    "exp": [3, 3, 0],
}
narrative_detail_weights_3 = {
    "org": [2, 4, 2, 1],
    "con": [4, 1, 0, 3],
    "exp": [2, 3, 1],
}
narrative_detail_weights_4 = {
    "org": [3, 3, 2, 1],
    "con": [4, 1, 0, 3],
    "exp": [2, 3, 2],
}
narrative_detail_weights_by_grade = {
    "초등_4학년" : narrative_detail_weights_1,
    "초등_5학년" : narrative_detail_weights_1,

    "초등_6학년" : narrative_detail_weights_2,
    "중등_1학년" : narrative_detail_weights_2,
    
    "중등_2학년" : narrative_detail_weights_3,
    "중등_3학년" : narrative_detail_weights_3,
    "고등_1학년" : narrative_detail_weights_3,

    "고등_2학년" : narrative_detail_weights_4,
    "고등_3학년" : narrative_detail_weights_4,
}

if __name__ == "__main__":
    # 사용 예시
    essay_type = "주장글"
    grade = "고등_1학년"

    category = essay_type_to_category[essay_type]
    if category == "논술형":
        detail_weights = argumentative_detail_weights_by_grade[grade]
    else:
        detail_weights = narrative_detail_weights_by_grade[grade]

    print("카테고리 :", category)
    print("구성 가중치 :", category_weights[category]["org"])
    print("구성 세부 가중치 :", detail_weights["org"])