# 안내
- 다음 파일들은 "제 3회 KRX 금융 언어모델 경진대회" 본선의 EVEN 팀 제출용 코드입니다.
- 코드의 내용은 하위 내용과 같으며, 관련 데이터는 huggingface에 업로드 하였습니다.

# 내용
## 모델 학습 및 병합
- data_utils.py : 관련 데이터를 로드하고, 전처리 등의 데이터 관련 코드가 담겨있습니다.
- parallel_model_utils.py : 모델 학습을 진행한 코드가 담겨있습니다. (SFT, Full Fine-tuning)
- parallel_model_merged.py : 학습을 완료한 모델을 huggingface에 업로드하고, merging 하는 등의 후처리 코드가 담겨있습니다.
- 모델 학습 코드.ipynb : 위 모듈들을 사용한 코드를 정리한 파일입니다.

## 데이터 제작
- data_gen.py : 데이터 생성을 위한 관련 함수 코드가 담겨잇습니다.
- 데이터 생성 코드.ipynb : 위 모듈을 사용한 코드를 정리한 파일입니다.
