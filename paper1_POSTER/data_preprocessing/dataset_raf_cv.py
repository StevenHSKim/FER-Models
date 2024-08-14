import torch.utils.data as data
import cv2
import pandas as pd
import os
import random
import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit

def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, subset='train', transform=None, basic_aug=False, random_state=None):
        self.subset = subset
        self.transform = transform
        self.raf_path = raf_path
        
        if random_state is not None:
            control_random_seed(random_state)

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        
        file_names = df.iloc[:, NAME_COLUMN].values
        labels = df.iloc[:, LABEL_COLUMN].values - 1

        # 데이터를 섞고 훈련, 검증, 테스트 데이터로 나눔
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_val_idx, test_idx = next(splitter.split(file_names))

        val_splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        train_idx, val_idx = next(val_splitter.split(train_val_idx))

        if self.subset == 'train':
            selected_idx = train_val_idx[train_idx]
        elif self.subset == 'val':
            selected_idx = train_val_idx[val_idx]
        elif self.subset == 'test':
            selected_idx = test_idx
        else:
            raise ValueError("Subset should be 'train', 'val', or 'test'")

        self.file_names = file_names[selected_idx]
        self.target = labels[selected_idx]

        self.file_paths = []
        for f in self.file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.target

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        sample = cv2.imread(path)
        target = self.target[idx]
        if self.subset == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                sample = self.aug_func[index](sample)

        if self.transform is not None:
            sample = self.transform(sample.copy())

        return sample, target

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)
