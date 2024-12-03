import os

from konlpy.tag import Okt
from kiwipiepy import Kiwi

okt = Okt()
kiwi = Kiwi(typos='basic') # 오타 교정기능 사용

# 불용어 사전 불러오기
stop_words_path = os.path.join(os.path.dirname(__file__), 'stopwords-ko.txt')
with open(stop_words_path, 'r', encoding='utf-8') as f:
    stop_words = f.readlines()
    stop_words = [word.replace('\n', '') for word in stop_words]

def stop_words_filter(sentence):
    word_tokens = okt.morphs(sentence)
    
    result = [word for word in word_tokens if word not in stop_words]
    return result

if __name__ == '__main__':
    sentence = '지금은 점심시간입니다. 그러니까 식사 맛있게 하세요.'
    
    print(f"원래 문장 : {sentence}")
    print(f"원래 문장 길이 : {len(sentence.replace(' ', ''))}")
    
    result = stop_words_filter(sentence)
    print(f"불용어 제거 : {' '.join(result)}")
    print(f"불용어 제거 문장 길이 : {len(''.join(result))}")
    
    print(f"단어 추출 : {okt.nouns(sentence)}")
    print(f"단어 개수 : {len(okt.nouns(sentence))}")
    

    # print(okt.phrases(sentence))