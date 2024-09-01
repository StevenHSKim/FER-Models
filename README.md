## 폴더 구조


```bash
FER-Models/
│
├── data/                              # 데이터 파일 및 관련 자료가 저장되는 디렉토리
│
├── evaluation/                        # 모델 평가 관련 파일들이 저장되는 디렉토리
│
├── preprocessing/                     # 데이터 전처리 스크립트들이 저장되는 디렉토리
│
├── requirements.txt                   # 프로젝트의 Python 패키지 의존성 목록을 명시한 파일
│
└── src/                               # 소스 코드가 저장되는 메인 디렉토리
    │
    ├── common/                        # 공통으로 사용되는 모듈들이 저장된 디렉토리
    │   ├── loss.py                    # 손실 함수 구현 파일
    │   ├── model_structure.py         # 모델 구조 정의 파일
    │   └── trainer.py                 # 모델 학습에 필요한 트레이너 파일
    │
    ├── configs/                       # 설정 파일들이 저장된 디렉토리
    │   ├── model/                     # 모델별 설정 파일들이 저장된 하위 디렉토리
    │   │   ├── DAN.yaml               # DAN 모델 설정 파일
    │   │   ├── DDAMFN.yaml            # DDAMFN 모델 설정 파일
    │   │   ├── POSTER.yaml            # POSTER 모델 설정 파일
    │   │   └── RAC.yaml               # RAC 모델 설정 파일
    │   └── overall.yaml               # 전체 프로젝트 설정 파일
    │
    ├── main.py                        # 프로젝트 메인 실행 파일
    │
    ├── models/                        # 모델 아키텍처들이 저장된 디렉토리
    │   ├── dan.py                     # DAN 모델 구현 파일
    │   ├── ddamfn.py                  # DDAMFN 모델 구현 파일
    │   ├── poster.py                  # POSTER 모델 구현 파일
    │   ├── pretrain/                  # 사전 학습된 모델 관련 파일들이 저장된 하위 디렉토리
    │   │   ├── MixedFeatureNet.py     # MixedFeatureNet 모델 구현 파일
    │   │   ├── hyp_crossvit.py        # Hyp_CrossViT 모델 구현 파일
    │   │   ├── ir50.py                # IR50 모델 구현 파일
    │   │   └── mobilefacenet.py       # MobileFaceNet 모델 구현 파일
    │   └── rac.py                     # RAC 모델 구현 파일
    │
    ├── utils/                         # 유틸리티 모듈들이 저장된 디렉토리
    │   ├── configurator.py            # 설정 파일 처리 유틸리티
    │   ├── data_utils.py              # 데이터 관련 유틸리티 함수들
    │   ├── dataloader.py              # 데이터 로딩 관련 파일
    │   ├── dataset.py                 # 데이터셋 처리 관련 파일
    │   ├── logger.py                  # 로깅 관련 유틸리티 파일
    │   ├── metrics.py                 # 평가 지표 관련 파일
    │   ├── run_experiments.py         # 실험 실행 유틸리티 파일
    │   └── utils.py                   # 기타 유틸리티 함수들
    │
    └── 실행파일/                        # 모델 실행 파일들이 저장된 디렉토리
        ├── dan_실행.py                # DAN 모델 실행 파일
        ├── ddamfn_실행.py             # DDAMFN 모델 실행 파일
        └── poster_실행.py             # POSTER 모델 실행 파일

```




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

- [RAC 논문 링크](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Leave+No+Stone+Unturned%3A+Mine+Extra+Knowledge+for+Imbalanced+Facial+Expression+Recognition&btnG=)
- [RAC Github 링크](https://github.com/zyh-uaiaaaa/Mine-Extra-Knowledge?tab=readme-ov-file)
