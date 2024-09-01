'''
데이터셋 분할 및 Dataset 클래스 제공
'''

import os
import random
import pandas as pd
import torch.utils.data as data
from PIL import Image
from sklearn.model_selection import ShuffleSplit

from data_utils import flip_image, add_gaussian_noise


def shuffle_split_data(file_paths, labels, config):
    """
    random state를 기반으로 shuffle split을 진행하는 함수
    """
    ss = ShuffleSplit(n_splits=config['iterations'], test_size=config['test_size'], random_state=config['random_state'])
    splits = list(ss.split(file_paths, labels))
    return splits


class RafDataSet(data.Dataset):
    """
    RAFDB 데이터셋을 위한 Dataset 클래스
    """
    def __init__(self, raf_path, labels, config, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        # augmentation 부분
        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

        self.raf_path = raf_path
        self.labels = labels


    def __len__(self):
        """
        데이터셋의 총 아이템 수를 반환
        """
        return len(self.file_paths)
  
    
    def get_labels(self):
        """
        모든 레이블을 반환
        """
        return self.labels


    def __getitem__(self, idx):
        """
        데이터셋에서 한 아이템을 가져오기
        """
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.basic_aug and random.uniform(0, 1) > 0.5:
            index = random.randint(0, 1)
            image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


    def load_raf_data(datapath):
        """
        RAF 데이터셋을 로드
        """
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(datapath, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)

        file_names = df.iloc[:, NAME_COLUMN].values
        labels = df.iloc[:, LABEL_COLUMN].values - 1
        file_paths = [os.path.join(datapath, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") for f in file_names]

        return file_paths, labels