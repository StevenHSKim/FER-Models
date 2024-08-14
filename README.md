## 파일 설명
**'rafdb_crossval.py' :** 반복 실험 코드
>  1. train, test 과정이 모두 포함됨
>  2. 모든 모델에서 하이퍼 파라미터가 통일됨
>  3. ShuffleSplit으로 10회 반복 실험이 포함됨
>  4. 모델 시드 고정됨     

'rafdb_train.py', 'rafdb_test.py': 구현 및 논문 결과 재현을 위해 사용한 코드
>  1. train, test 과정이 각각 존재함
>  2. 모델마다 각 논문에서 제시한 하이퍼 파라미터가 사용되었으며 통일되지 않음

(rafdb 이외에 affctnet 관련 코드는 원본 그대로의 상태로, 수정하지 않았습니다.)

## 원본 코드 링크
1. POSTER: <https://github.com/zczcwh/POSTER>
2. DAN: <https://github.com/yaoing/DAN>
3. DDAMFN: <https://github.com/SainingZhang/DDAMFN>
