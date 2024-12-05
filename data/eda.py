import os

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
