## 폴더 구조


```bash
FER-Models/                            
├── checkpoints/                    # 모델 체크포인트를 저장하는 디렉터리
├── config/                         # 데이터셋 및 설정 파일을 포함하는 디렉터리
│   ├── __init__.py             
│   └── dataset_configs.py          # 데이터셋 관련 설정 파일
├── dataset/                        # 원본 데이터 및 전처리된 데이터를 저장하는 디렉터리
├── environment_all.yaml            # 전체 환경 설정에 필요한 모든 패키지 목록
├── environment_essential.yaml      # 핵심적으로 필요한 패키지 목록
└── src/                            # 소스 코드 디렉터리
    ├── common/                     # 공통 모듈 (손실 함수, 트레이너 등)
    │   ├── __init__.py         
    │   ├── loss.py                 # 손실 함수 정의 파일
    │   └── trainer.py              # 학습 과정 정의 및 실행 파일
    ├── main.py                     # 프로젝트의 메인 실행 파일
    ├── models/                     # 모델 아키텍처 관련 파일
    │   ├── __init__.py         
    │   ├── adadf/                  # AdaDF 모델 관련 모듈
    │   │   ├── __init__.py     
    │   │   ├── adadf.py            # AdaDF 모델 정의
    │   │   ├── auto_augment.py     # 자동 데이터 증강 관련 코드
    │   │   └── resnet18.py         # ResNet18 모델 코드
    │   ├── dan/                    # DAN 모델 관련 모듈
    │   │   ├── __init__.py     
    │   │   └── dan.py              # DAN 모델 정의
    │   ├── ddamfn/                 # DDAMFN 모델 관련 모듈
    │   │   ├── __init__.py     
    │   │   ├── DDAM.py             # DDAM 모델 정의
    │   │   └── MixedFeatureNet.py  # MFN 모델 정의
    │   ├── poster/                 # Poster 관련 모듈
    │   │   ├── __init__.py     
    │   │   ├── crossvit.py         # CrossViT 모델 정의
    │   │   ├── ir50.py             # 백본 IR-50 모델 코드
    │   │   ├── mobilefacenet.py    # 백본 MobileFaceNet 모델 코드
    │   │   └── poster.py           # Poster 모델 정의
    │   └── pretrain_weights/       # 사전 학습된 가중치 저장 디렉터리
    └── utils/                      # 유틸리티 코드 모음
        ├── __init__.py             
        ├── data_utils.py           # 데이터 관련 유틸리티
        ├── dataloader.py           # 데이터 로더 정의
        ├── dataset.py              # 데이터셋 처리 정의
        ├── run_experiments.py      # 실험 실행 및 관리 코드
        └── utils.py                # 기타 유틸리티 함수 정의

```

<br>

## 논문 정보 및 링크

| **모델 이름** | **논문 제목** | **Venue** | **논문 링크** | **Github 링크** |
|:---------------:|---------------|----------|:---------------:|:-----------------:|
| POSTER | "A pyramid cross-fusion transformer network for facial expression recognition" | ICCV Workshop (AMFG) 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Zheng%2C+Ce%2C+Matias+Mendieta%2C+and+Chen+Chen.+%22Poster%3A+A+pyramid+cross-fusion+transformer+network+for+facial+expression+recognition.%22+Proceedings+of+the+IEEE%2FCVF+International+Conference+on+Computer+Vision.+2023.&btnG=) | [Github](https://github.com/zczcwh/POSTER) |
| DAN | "Distract your attention: Multi-head cross attention network for facial expression recognition" | Biomimetics 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Wen%2C+Zhengyao%2C+et+al.+%22Distract+your+attention%3A+Multi-head+cross+attention+network+for+facial+expression+recognition.%22+Biomimetics+8.2+%282023%29%3A+199.&btnG=) | [Github](https://github.com/yaoing/DAN) |
| DDAMFN | "A dual-direction attention mixed feature network for facial expression recognition" | Electronics 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Zhang%2C+Saining%2C+et+al.+%22A+dual-direction+attention+mixed+feature+network+for+facial+expression+recognition.%22+Electronics+12.17+%282023%29%3A+3595.&btnG=) | [Github](https://github.com/SainingZhang/DDAMFN) |
| LNSU-Net | "Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition" | NeurIPS 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Leave+No+Stone+Unturned%3A+Mine+Extra+Knowledge+for+Imbalanced+Facial+Expression+Recognition&btnG=) | [Github](https://github.com/zyh-uaiaaaa/Mine-Extra-Knowledge?tab=readme-ov-file) |
| Ada-DF | "A Dual-Branch Adaptive Distribution Fusion Framework for Real-World Facial Expression Recognition" | ICASSP 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=A+Dual-Branch+Adaptive+Distribution+Fusion+Framework+for+Real-World+Facial+Expression+Recognition.&btnG=) | [Github](https://github.com/taylor-xy0827/Ada-DF) |
| POSTER++ | "POSTER++: A simpler and stronger facial expression recognition network" | Pattern Recognit. 2024 | [Paper](https://www.sciencedirect.com/science/article/pii/S0031320324007027) | [Github](https://github.com/talented-q/poster_v2) |
| MFER | "Multiscale Facial Expression Recognition Based on Dynamic Global and Static Local Attention" | IEEE Trans. Affect. Comput. 2024 | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10678884) | [Github](https://github.com/XuJ1E/MFER/?tab=readme-ov-file) |
| GSDNet | "A gradual self distillation network with adaptive channel attention for facial expression recognition" | Appl. Soft. Comput. 2024 | [Paper](https://www.sciencedirect.com/science/article/pii/S1568494624005362) | [Github](https://github.com/Emy-cv/GSDNet) |


<br>

## 데이터셋
| **데이터셋 이름** | **이미지 개수** | **공식 홈페이지** | **취득 방법** | **데이터셋 다운로드 링크** |
|:---------------:|:----------:|:---------------:|---------------|:---------------:|
| RAFDB | 15339 | [Homepage](http://www.whdeng.cn/RAF/model1.html#dataset) | [MTCNN](https://github.com/foamliu/Face-Alignment)을 이용하여 얼굴을 정렬 완료한 데이터셋을 다운로드 받아서 사용 하였음 | [Google Drive](https://drive.google.com/file/d/1GiVsA5sbhc-12brGrKdTIdrKZnXz9vtZ/view) |
| FER2013 | 35887 | [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | 공식 Kaggle 링크에서 `icml_face_data.csv`를 다운로드 받은 뒤 "emotion"과 "pixel" 열만 남긴 `fer2013_modified`를 생성하여 사용하였음 | [Google Drive](https://drive.google.com/drive/folders/1-mGIAchWBUEhgmIKT36PrvQ1-LXl3Y5n?usp=sharing) |
| FERPlus | 35711 | [Github](https://github.com/Microsoft/FERPlus) | 공식 Github 링크에서 `fer2013new.csv`를 다운로드 받고, 직접 정의한 [label 생성 코드](https://github.com/StevenHSKim/FERPlus_Vote_To_Label)를 통해 `FERPlus_Label_modified.csv`를 생성하였음. [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)의 pixel값을 공식 Github의 `generate_training_data.py`로 png로 바꾸어 `FERPlus_Image`를 생성하였음 | [Google Drive](https://drive.google.com/drive/folders/1n73_68Zq4aa0KBImIANHhiSJMg6j2zVV?usp=sharing) |
| ExpW | 90560 | [Homepage](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html) | 홈페이지에서 다운로드 받은 뒤, [MTCNN](https://github.com/foamliu/Face-Alignment)을 이용하여 얼굴을 직접 정렬 하였음 | [Google Drive](https://drive.google.com/drive/folders/1jNmC5RWqyBFvFsTHnWpg-cti0kpWxEi0?usp=sharing) |
| SFEW2.0 | 1634 | [Homepage](https://users.cecs.anu.edu.au/~few_group/AFEW.html) | 홈페이지를 통해 저자에게 데이터셋을 요청한 뒤, `_Aligned_Face` 데이터셋을 다운로드 받아 얼굴이 아닌 이미지는 직접 삭제한 뒤에 사용하였음 | [Google Drive](https://drive.google.com/drive/folders/1FuhcMW5LXaaWe8sKoGizQW78s04VcOHW?usp=sharing) |
| CK+ | 981 | [Homepage](https://www.jeffcohn.net/Resources/), [Kaggle](https://www.kaggle.com/datasets/shuvoalok/ck-dataset) | 각 동영상의 마지막 3개의 프레임을 캡처한 데이터셋을 다운로드 받아서 사용하였음 | [Google Drive](https://drive.google.com/drive/folders/1kuT6zQhZtyBPgTB4UqNJWq0ZBLAooTfA?usp=sharing) |

<br>

### RAFDB
- RAFDB 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/raf-basic/
    EmoLabel/
        list_patition_label.txt
    Image/aligned/
        train_00001_aligned.jpg
        test_0001_aligned.jpg
        ...
```

### FER2013
- FER2013 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/FER2013/
    fer2013_modified.csv
```

### FERPlus
- FERPlus 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/FERPlus/
    FERPlus_Label_modified.csv
    FERPlus_Image/
        fer0000000.png
        fer0000001.png
        ...
```

### ExpW
- ExpW 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/ExpW/
    label/
        label.lst
    aligned_image/
        afraid_African_214.jpg
        afraid_american_190.jpg
        ...
```

### SFEW2.0
- SFEW2.0 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/SFEW2.0/
    sfew_2.0_labels.csv
    sfew2.0_images/
        image_000000.png
        image_000001.png
        ...
```

### CK+
- CK+ 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/CKPlus/
    ckplus_labels.csv
    ckplus_images/
        image_000000.png
        image_000001.png
        ...
```

<br>

## 사전학습 모델

| **모델 이름** | **사용되는 사전학습 모델 다운로드 링크** | **설명** |
|:---------------:|:----------:|----------|
| POSTER | [ir50 & mobilefacenet](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq) | improved resnet-50(이미지 특징 추출 백본)과 mobilefacenet(랜드마크 특징 추출 백본) |
| DAN | [resnet18](https://drive.google.com/file/d/1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n/view) | msceleb 데이터셋으로 학습시킨 resnet-18 백본 |
| DDAMFN | [mixedfeaturenet](https://github.com/SainingZhang/DDAMFN/blob/main/pretrained/MFN_msceleb.pth) | msceleb 데이터셋으로 학습시킨 mixedfeaturenet 백본 |
| LNSUNet | [swin transformer](https://drive.google.com/file/d/1GiVsA5sbhc-12brGrKdTIdrKZnXz9vtZ/view) | swin transformer 백본 |
| Ada-DF | [resnet18](https://drive.google.com/file/d/1ByvxPD9QkmWZDWtTmDQ5ta1MiAkXt22T/view) | msceleb 데이터셋으로 학습시킨 resnet-18 백본 |
| POSTER++ | [ir50](https://drive.google.com/file/d/17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt/view), [mobilefacenet](https://drive.google.com/file/d/1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX/view) | improved resnet-50(이미지 특징 추출 백본)과 mobilefacenet(랜드마크 특징 추출 백본) |
| MFER | [resnet18](https://drive.google.com/file/d/1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n/view) | msceleb 데이터셋으로 학습시킨 resnet-18 백본 |
| GSDNet | [resnet50](https://drive.google.com/drive/folders/1OUrrHPYRDneS5OcE6sk8PwbfP_zhmvmJ) | msceleb 데이터셋으로 학습시킨 resnet-50 백본 |

<br>

위 사전학습 모델을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
models/pretrain/
    ir50.pth                            # improved resnet50
    mobilefacenet_model_best.pth        # mobilefacenet
    resnet18_msceleb.pth                # resnet18
    start_0.pth                         # swin transformer
    vgg_msceleb_resnet50_ft_weight.pkl  # resnet50
```
