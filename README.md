## 파일 설명
**'rafdb_crossval.py' :** 반복 실험 코드
> 1. train, test 과정이 모두 포함됨
> 2. 모든 모델에서 하이퍼 파라미터가 통일됨
> 3. ShuffleSplit으로 10회 반복 실험이 포함됨
> 4. 모델 시드 고정됨
> 5. (반복 실험을 위해 새로 생성함) 

'rafdb_train.py', 'rafdb_test.py' : 모델 구현 및 논문 결과 재현을 위해 사용한 코드
> 1. train, test 과정이 각각 존재함
> 2. 모델마다 각 논문에서 제시한 하이퍼 파라미터가 사용되었으며 통일되지 않음
> 3. (원본 코드에 이미 존재하였음)

<br>

- 사용된 하이퍼 파라미터들은 위 세가지 파일들의 가장 상단에 `parse_args` 함수에서 확인하실 수 있습니다. 추가 인자 입력 없이 `default`에 할당된 값들만을 사용하였습니다.
<br>

- 데이터셋 (rafdb, affectnet, ferplus 등)별로 train,test 파일을 분리해두는 FER 분야 코드의 경향을 그대로 유지하였습니다.
- rafdb 이외에 affectnet 관련 코드는 원본 그대로의 상태로, 수정하지 않았습니다.

<br>

## 논문 정보 및 링크
### POSTER
> "Poster: A pyramid cross-fusion transformer network for facial expression recognition"<br>
> ICCV Workshop (AMFG) 2023

- [POSTER 논문 링크](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Zheng%2C+Ce%2C+Matias+Mendieta%2C+and+Chen+Chen.+%22Poster%3A+A+pyramid+cross-fusion+transformer+network+for+facial+expression+recognition.%22+Proceedings+of+the+IEEE%2FCVF+International+Conference+on+Computer+Vision.+2023.&btnG=)
- [POSTER Github 링크](https://github.com/zczcwh/POSTER)

  
### DAN
> "Distract your attention: Multi-head cross attention network for facial expression recognition"<br>
> Biomimetics 2023

- [DAN 논문 링크](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Wen%2C+Zhengyao%2C+et+al.+%22Distract+your+attention%3A+Multi-head+cross+attention+network+for+facial+expression+recognition.%22+Biomimetics+8.2+%282023%29%3A+199.&btnG=)
- [DAN Github 링크](https://github.com/yaoing/DAN)

  
### DDAMFN
> "A dual-direction attention mixed feature network for facial expression recognition"<br>
> Electronics 2023

- [DDAMFN 논문 링크](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Zhang%2C+Saining%2C+et+al.+%22A+dual-direction+attention+mixed+feature+network+for+facial+expression+recognition.%22+Electronics+12.17+%282023%29%3A+3595.&btnG=)
- [DDAMFN Github 링크](https://github.com/SainingZhang/DDAMFN)


### RAC
> "Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition"<br>
> NeurIPS 2023

- [DDAMFN 논문 링크](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Leave+No+Stone+Unturned%3A+Mine+Extra+Knowledge+for+Imbalanced+Facial+Expression+Recognition&btnG=)
- [DDAMFN Github 링크](https://github.com/zyh-uaiaaaa/Mine-Extra-Knowledge?tab=readme-ov-file)
