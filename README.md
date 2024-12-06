# 개요
> [AI-Hub의 "에세이 글 평가 데이터"](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=545)를 사용하여, 자동화된 에세이 평가 시스템을 개발하는 프로젝트입니다.
> 해당 프로젝트는 [하이퍼러닝](https://www.hyperlearning.kr)의 채용과정 중 과제 전형으로 진행되었습니다.

## 모델 선정
- [klue/roberta-base](https://huggingface.co/klue/roberta-base) 모델을 사용했습니다.
- 해당 모델은 BERT 기반의 `RoBERTa`모델로 한국어에 특화되어 준수한 성능을 보입니다.
- `klue/roberta-large`모델의 성능이 더욱 좋으므로 추후에 모델 변경으로 성능 향상을 기대할 수 있습니다.

## 사용한 패키지
- 한국어 불용어 : [spikeekips/stopwords-ko.txt](https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a)를 `data/stopwords-ko.txt`에 추가했습니다. 하지만, 여기에서는 사용하지 않았습니다.
- [한국어 형태소 분석기 Kiwi](https://github.com/bab2min/kiwi)의 [Python용 Kiwi 패키지 kiwipiepy](https://github.com/bab2min/kiwipiepy)를 사용했습니다.
- [KoNLPy](https://konlpy.org/ko/v0.6.0/)는 `JPype1`와 같은 Java관련 설정이 추가적으로 필요하므로, 간단한 `kiwipiepy`를 사용하였습니다.

## 데이터 처리
- 해당 모델은 최대 입력 토큰의 수가 512이므로, 긴 문장에 대해서는 `Sliding Window`를 적용하였습니다.

### 개발 환경
- OS : [Windows 11, macOS 15.1]
- Python : 3.11
# 실행 방법
## 환경설정
- [uv](https://docs.astral.sh/uv/)로 가상환경 설정을 하였습니다.
- 아래와 같이 기존의 방법대로도 사용이 가능합니다.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CUDA 사용방법
```bash
# uv 설치 후 사용 예시
uv sync --extra cu124
```

## 실행과정
1. AI-Hub에서 데이터를 다운로드 받아 디렉토리에 추가합니다.
2. 운영체제에 따라 윈도우에서는 `unzip-all.ps1`을, mac/linux에서는 `unzip-all.sh`을 실행하여 압축을 해제합니다.
3. 필요한 패키지를 설치하고 `python data/preprocessing.py`를 실행하여 데이터 CSV 파일을 생성합니다.
4. `python data/eda.py`로 데이터의 특성을 확인할 수 있습니다.
5. `python src/train.py`로 학습을 진행 할 수 있습니다.