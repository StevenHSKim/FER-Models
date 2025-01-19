## 폴더 구조


```bash
FER-Models/                            
│
├── data/                              # 데이터 파일 및 관련 자료가 저장되는 디렉토리
│
├── requirements.yaml                  # Python 패키지 dependencies 목록을 명시한 파일
│
└── src/                               # 소스 코드가 저장되는 메인 디렉토리
    │
    ├── common/                        # 공통으로 사용되는 모듈들이 저장된 디렉토리
    │   ├── loss.py                    # 다양한 사용자 정의 손실 함수 정의
    │   └── trainer.py                 # 모델 학습에 필요한 트레이너 파일
    │
    ├── configs/                       # 설정 파일들이 저장된 디렉토리
    │   ├── model/                     # 모델별 설정 파일들이 저장된 하위 디렉토리
    │   │   ├── DAN.yaml               # DAN 모델 설정 파일
    │   │   ├── DDAMFN.yaml            # DDAMFN 모델 설정 파일
    │   │   └── POSTER.yaml            # POSTER 모델 설정 파일
    │   └── overall.yaml               # 공통된 설정 파일
    │
    ├── main.py                        # 메인 실행 파일
    │
    ├── models/                        # 모델 아키텍처들이 저장된 디렉토리
    │   ├── dan.py                     # DAN 모델 구현 파일
    │   ├── ddamfn.py                  # DDAMFN 모델 구현 파일
    │   ├── poster.py                  # POSTER 모델 구현 파일
    │   └── pretrain/                  # 사전 학습된 모델 관련 파일들이 저장된 하위 디렉토리
    │       ├── MixedFeatureNet.py     # MixedFeatureNet 모델 구현 파일
    │       ├── hyp_crossvit.py        # Hyp_CrossViT 모델 구현 파일
    │       ├── ir50.py                # IR50 모델 구현 파일
    │       └── mobilefacenet.py       # MobileFaceNet 모델 구현 파일
    │
    └── utils/                         # 유틸리티 모듈들이 저장된 디렉토리
        ├── configurator.py            # 설정 파일 처리 유틸리티
        ├── data_utils.py              # 데이터 관련 유틸리티 함수들
        ├── dataloader.py              # 데이터 로딩 관련 파일
        ├── dataset.py                 # 데이터셋 처리 관련 파일
        ├── logger.py                  # 로깅 관련 유틸리티 파일
        ├── metrics.py                 # 평가 지표 관련 파일
        ├── run_experiments.py         # 실험 실행 유틸리티 파일
        └── utils.py                   # 기타 유틸리티 함수들
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


### LNSU-Net
> "Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition"<br>
> NeurIPS 2023

- [LNSU-Net 논문 링크](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Leave+No+Stone+Unturned%3A+Mine+Extra+Knowledge+for+Imbalanced+Facial+Expression+Recognition&btnG=)
- [LNSU-Net Github 링크](https://github.com/zyh-uaiaaaa/Mine-Extra-Knowledge?tab=readme-ov-file)


### Ada-DF
> "A Dual-Branch Adaptive Distribution Fusion Framework for Real-World Facial Expression Recognition"<br>
> ICASSP 2023

- [Ada-DF 논문 링크](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=A+Dual-Branch+Adaptive+Distribution+Fusion+Framework+for+Real-World+Facial+Expression+Recognition.&btnG=)
- [Ada-DF Github 링크](https://github.com/taylor-xy0827/Ada-DF)


### POSTER++
> "POSTER++: A simpler and stronger facial expression recognition network"<br>
> Pattern Recognit. 2024

- [POSTER++ 논문 링크](https://www.sciencedirect.com/science/article/pii/S0031320324007027)
- [POSTER++ Github 링크](https://github.com/talented-q/poster_v2)


### MFER
> "Multiscale Facial Expression Recognition Based on Dynamic Global and Static Local Attention"<br>
> IEEE Trans. Affect. Comput. 2024

- [MFER 논문 링크](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10678884)
- [MFER Github 링크](https://github.com/XuJ1E/MFER/?tab=readme-ov-file)


### GSDNet
> "A gradual self distillation network with adaptive channel attention for facial expression recognition"<br>
> Appl. Soft. Comput. 2024

- [GSDNet 논문 링크](https://pdf.sciencedirectassets.com/272229/1-s2.0-S1568494624X00101/1-s2.0-S1568494624005362/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCICQFCchd0sby4HxcuaQJAxsyqg%2FhPExv%2FwxiOxtFoMZVAiEAmL6I67PPYelV4hKFlr6bik3Bh3Z%2BPbTnJpwruTSc2E4qvAUIi%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDGk4KXTlHzsNmbdmUSqQBQUi0TJso5AaNn6hUev8iDIUCVXOpDn9t2OC4fBE2JHhSH6HEhL0jc2IlhzRdNB7Ha0950m7X14EISBlF0HEjVri2%2FenN2woHqNLZT6yGD0Ug8cUYPhhyWMN3f7QiobiYIt3QRgpzYWkceRKYRsbZdROVR1eeKh%2FBevXnjLMvDiQaD2NVWdCBFOGGhuTPo%2BPWYs8AmV3dHoeXugGwG2zjhHPW%2FVGrlSeUkklbEFgSnR4wScyMraYEdto1BmYY0fmH%2FXn2sNuPp0Cl3y1GauDanxF9mM8nnPWwcyBhGi9Z9f7bVobZk77DfK6arOtIwxYFORopI4%2Bm5Nj4WXbBFJhkkmHnhe%2FFzEryNKeM%2FrDwKlpmxEoKIs4T54yicNMLJGHqTSOG0TXvlx6ZFNOEweRBGztaylX1wA2f%2B1tth0a%2BsIhdXfnDzJEvpOj1VfvNe6x9Rgq9I%2FvfjLJukojWz%2BYn6l80bSCdVp89WgK47HvD0Q%2BH4HP6hQUwFS%2FtvObidbw58KNE65A%2Fs5Q10k%2Frd9p3%2BN8LLmQRvgNAWCpmh7ZruLJSaBUXNbLCZOR4oW1qywIvink7BN9nRs31GFAMLYMZBQ6f%2BXr4%2FPnz2yC4HBsWWWF5rC%2FwI58xNgVkKwV4oobxm2iAvR3u%2FH%2B0nve9iUx2a7utSZWseyCG6uCQCUWk9ElgjHOmzuU891Kl2mvJcOeFiiyJhQWcojqiKJPmmLo9J1NIE14aZbIH7wYsfpkXlLbaghG%2FLKKkHRT5AHaLLUnLTR%2BuDUjwTVn3W0DUU5r0LvrINLqxNHC1f4bhyUteQ3cLwyqnmqCroHechweQvrFYj1JuunQ6XUDKHFME%2F%2FFocHC1orBd768nRxa4N2kCq5YMISXs7wGOrEBbScr1M9UcnbdlJnx0qZFO8%2FoQPnTVkAexceqmv30b%2BSo%2Bs69hotjf83Cxez7lFaMzz%2BhiqkFa914N%2FDrBGoeQE6%2FUaUpF5H8Mq%2FfG8zaJlu8sXfTPdwSJL%2BCzYIiVngBG0Wd9j4RGePwiCiX8VMyQ1NoTHxO8Z00STVOi2ySBrDastiEJkjLYtWac0PP%2F1m2G19DKiFtb4lPMTQGe0gqt6GsJ6hV3J86NdZRSkr4C09u&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250119T103707Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6KOPIJKO%2F20250119%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8f2dad077d509ef51c611e8d264cd13a13b66a7f26f562fe2ba1ce911d3aad8e&hash=06c3022c64e1a5ce216c92b80c71f9f81f70a01bccd2554ca62610b1c1e264db&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1568494624005362&tid=spdf-a5db43b8-d251-4c30-95bd-3858dbdeb00b&sid=29f211986649824dc4897c0-a71db1183834gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=11145c5f57515c5e565800&rr=90463007e888ea9b&cc=kr)
- [GSDNet Github 링크](https://github.com/Emy-cv/GSDNet/blob/main/GSD-Net/train.py)


<br>

## 데이터셋
- [RAF-DB](http://www.whdeng.cn/raf/model1.html)
- [FERPlus](https://github.com/Microsoft/FERPlus)
- [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [ExpW](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html)
- [SFEW2.0](https://users.cecs.anu.edu.au/~few_group/AFEW.html)
- [CK+](https://www.jeffcohn.net/Resources/)
