import os

import pandas as pd
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

kiwi = Kiwi(typos='basic') # 오타 교정기능 사용

# Kiwi의 불용어 사전을 사용
# stopwords = Stopwords()

# 불용어 사전 불러오기
stop_words_path = os.path.join(os.path.dirname(__file__), 'stopwords-ko.txt')
with open(stop_words_path, 'r', encoding='utf-8') as f:
    stop_words = f.readlines()
    stop_words = [word.replace('\n', '') for word in stop_words]

def stop_words_filter(sentence: str) -> str:
    word_tokens = kiwi.tokenize(sentence)
    word_tokens = [word.form for word in word_tokens]
    result = ''.join([word for word in word_tokens if word not in stop_words])
    result = kiwi.join(kiwi.tokenize(kiwi.space(result)))
    return result

# 명사 추출
def extract_nouns(sentence: str) -> list[str]:
    noun_tags = ["NNG", "NNP", "NNB", "NP", "NR", "SL"]
    word_tokens = kiwi.tokenize(sentence)
    nouns = [word.form for word in word_tokens if word.tag in noun_tags]
    return nouns

# 고유 명사 추출
def extract_proper_nouns(sentence: str) -> list[str]:
    word_tokens = kiwi.tokenize(sentence)
    proper_nouns = [word.form for word in word_tokens if word.tag == "NNP"]
    return proper_nouns

if __name__ == '__main__':
    # 사용 예시
    # 1개의 고유명사, 1개의 불용어가 포함된 문장
    sentence = '안녕하세요. 제 이름은 구희찬이고요. 지금은 점심시간입니다. 그러니까 식사 맛있게 하세요.'
    
    print(f"원래 문장 : {sentence}")
    print(f"원래 문장 길이 : {len(sentence.replace(' ', ''))}\n")
    
    result = stop_words_filter(sentence)
    print(f"불용어 제거 : {result}")
    print(f"불용어 제거 문장 길이 : {len(result)}\n")
    
    nouns = extract_nouns(sentence)
    print(f"명사 추출 : {nouns}")
    print(f"명사 개수 : {len(nouns)}\n")

    proper_nouns = extract_proper_nouns(sentence)
    print(f"고유 명사 추출 : {proper_nouns}")
    print(f"고유 명사 개수 : {len(proper_nouns)}")

    from matplotlib import pyplot as plt

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data.csv'))
    
    # 데이터 개수
    print(f"\n데이터 개수 : {len(df):,}")

    # 단어 수 분포
    df['word_count'] = df['essay'].apply(lambda x: len(x.split()))
    print(f"\n총 단어 수 : {df['word_count'].sum():,}")
    print(f"평균 단어 수 : {df['word_count'].mean()}")
    print(f"최대 단어 수 : {df['word_count'].max()}")
    print(f"최소 단어 수 : {df['word_count'].min()}")
    plt.hist(df['word_count'], bins=100)
    plt.title('Word Count Distribution')
    plt.show()

    # 문장 수 분포
    df['sentence_count'] = df['essay'].apply(lambda x: x.count('.'))
    print(f"\n총 문장 수 : {df['sentence_count'].sum():,}")
    print(f"평균 문장 수 : {df['sentence_count'].mean()}")
    print(f"최대 문장 수 : {df['sentence_count'].max()}")
    print(f"최소 문장 수 : {df['sentence_count'].min()}")
    plt.hist(df['sentence_count'], bins=100)
    plt.title('Sentence Count Distribution')
    plt.show()

    # 문단 수 분포
    counter = {1: 37863, 2: 783, 3: 435, 4: 254, 5: 119, 6: 51, 7: 35, 8: 14, 10: 11, 9: 9, 11: 4, 12: 4, 13: 3, 14: 3, 15: 2, 16: 1}
    plt.bar(counter.keys(), counter.values())
    plt.title('Paragraph Count Distribution')
    plt.show()

    # 고유 단어수
    words = []
    for essay in df['essay']:
        words += essay.split()
    unique_words = set(words)
    print(f"\n고유 단어 수 : {len(unique_words):,}")

    """
    아래의 코드는 실행 시간이 오래 걸립니다.
    """
    # # 명사 수 분포
    # df['nouns'] = df['essay'].apply(extract_nouns)
    # df['noun_count'] = df['nouns'].apply(len)
    # print(f"명사 수 : {df['noun_count'].sum()}")
    # plt.hist(df['noun_count'], bins=50)
    # plt.title('Noun Count Distribution')
    # plt.show()

    # # 고유 명사 수
    # proper_nouns = []
    # for essay in df['essay']:
    #     proper_nouns += extract_proper_nouns(essay)
    # unique_proper_nouns = set(proper_nouns)
    # print(f"총 고유 명사 수 : {len(proper_nouns)}")
    # print(f"고유 명사 수 : {len(unique_proper_nouns)}")

