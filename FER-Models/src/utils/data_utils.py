'''
이미지 데이터의 변환 및 증강을 관리하는 유틸리티 모듈
'''

import cv2
import numpy as np
from torchvision import transforms


class DataTransforms:
    """
    다양한 모델 유형에 맞게 이미지 데이터 변환을 설정하는 클래스
    """
    def __init__(self, model_type, train=True):
        self.model_type = model_type
        self.train = train


    def get_transforms(self):
        """
        모델 유형에 따라 적절한 데이터 변환을 반환
        """
        if self.model_type == 'POSTER':
            return self._get_poster_transforms()
        elif self.model_type == 'DAN':
            return self._get_dan_transforms()
        elif self.model_type == 'DDAMFN':
            return self._get_ddamfn_transforms()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


    def _get_poster_transforms(self):
        """
        POSTER 모델에 사용되는 데이터 변환을 반환
        """
        if self.train:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(scale=(0.02, 0.1)),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _get_dan_transforms(self):
        """
        DAN 모델에 사용되는 데이터 변환을 반환
        """
        if self.train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.RandomRotation(20),
                    transforms.RandomCrop(224, padding=32)
                ], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(scale=(0.02, 0.25)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]) 

    def _get_ddamfn_transforms(self):
        """
        DDAMFN 모델에 사용되는 데이터 변환을 반환
        """
        if self.train:
            return transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.RandomRotation(5),
                    transforms.RandomCrop(112, padding=8)
                ], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(scale=(0.02, 0.25)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


### POSTER에서 이용하는 augmentation 함수들

def add_gaussian_noise(image_array, mean=0.0, var=30):
    """
    이미지에 가우시안 노이즈를 추가하는 함수
    """
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped


def flip_image(image_array):
    """
    이미지를 수평으로 뒤집는 함수
    """
    return cv2.flip(image_array, 1)