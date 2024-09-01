'''
데이터셋 로드, 분할 및 DataLoader 설정을 관리
'''


import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import RafDataSet
from data_utils import DataTransforms


def load_dataset_list(datapath):
    """
    주어진 경로에서 데이터셋 목록을 로드
    """
    NAME_COLUMN = 0
    LABEL_COLUMN = 1
    file_names = df.iloc[:, NAME_COLUMN].values
    
    df = pd.read_csv(os.path.join(datapath, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
    file_paths = [os.path.join(datapath, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") for f in file_names]
    labels = df.iloc[:, LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    
    return file_paths, labels


def get_datasets(config, train_val_indices, test_indices):
    """
    Train, Validation, Test 데이터셋을 로드 및 분할
    """
    file_paths, labels = load_dataset_list(config['raf_path'])
    train_indices, val_indices = train_test_split(train_val_indices, test_size=config['val_size'], random_state=42)

    data_transforms = DataTransforms(model_type=config['model_type']) # 'model_type'이 들어와야하는지 'model'이 들어와야 하는지 헷갈리네
    
    train_dataset = RafDataSet(file_paths[train_indices], labels[train_indices], transform=data_transforms.get_transforms(train=True), basic_aug=True)
    val_dataset = RafDataSet(file_paths[val_indices], labels[val_indices], transform=data_transforms.get_transforms(train=False))
    test_dataset = RafDataSet(file_paths[test_indices], labels[test_indices], transform=data_transforms.get_transforms(train=False))
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size


class TrainDataLoader(torch.utils.data.DataLoader):
    """
    훈련용 DataLoader를 설정하는 클래스
    """
    def __init__(self, config, dataset, **kwargs):
        super().__init__(dataset, batch_size=config['batch_size'], num_workers=config['workers'], **kwargs)


class EvalDataLoader(torch.utils.data.DataLoader):
    """
    평가용 DataLoader를 설정하는 클래스
    """
    def __init__(self, config, dataset, **kwargs):
        super().__init__(dataset, batch_size=config['val_batch_size'], num_workers=config['workers'], **kwargs)
