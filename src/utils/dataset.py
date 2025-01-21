import os
import numpy as np
import pandas as pd
import cv2
import random
from PIL import Image
import torch.utils.data as data

from data_utils import poster_add_gaussian_noise, poster_flip_image
from config.dataset_configs import *

class Poster_UnifiedDataset(data.Dataset):
    def __init__(self, file_paths, labels, transform=None, basic_aug=False, dataset_type='rafdb'):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.basic_aug = basic_aug
        self.aug_func = [poster_flip_image, poster_add_gaussian_noise]
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        if self.dataset_type == 'fer2013':
            pixels = self.file_paths[idx].reshape(48, 48)
            image = pixels.astype(np.uint8)
            image = np.stack([image] * 3, axis=-1)
            image = cv2.resize(image, (224, 224))
        else:
            path = self.file_paths[idx]
            image = cv2.imread(path)
            if self.dataset_type == 'ferplus':
                image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)

        label = self.labels[idx]

        if self.basic_aug and random.uniform(0, 1) > 0.5:
            index = random.randint(0, 1)
            image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image.copy())

        return image, label
    
    
class DAN_UnifiedDataset(data.Dataset):
    def __init__(self, dataset_type, data_path, indices, transform=None, phase='train'):
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.transform = transform
        self.phase = phase
        self.config = DAN_DATASET_CONFIGS[dataset_type]

        if dataset_type == 'fer2013':
            df = pd.read_csv(os.path.join(data_path, self.config['label_file']))
            self.pixels = df['pixels'].values[indices]
            self.labels = df['emotion'].values[indices]
        elif dataset_type == 'rafdb':
            df = pd.read_csv(os.path.join(data_path, self.config['label_file']), 
                           sep=' ', header=None, names=['name', 'label'])
            self.file_names = df['name'].values[indices]
            self.labels = df['label'].values[indices] - 1  # RAF-DB는 1부터 시작
            self.file_paths = [os.path.join(data_path, self.config['image_dir'], 
                             f.split(".")[0] + "_aligned.jpg") for f in self.file_names]
        elif dataset_type == 'ferplus':
            df = pd.read_csv(os.path.join(data_path, self.config['label_file']))
            self.file_names = df['Image name'].values[indices]
            self.labels = df['label'].values[indices]
            if self.labels.min() > 0:  # 레이블이 1부터 시작하면 0부터 시작하도록 조정
                self.labels = self.labels - 1
            self.file_paths = [os.path.join(data_path, self.config['image_dir'], f) 
                             for f in self.file_names]
        elif dataset_type == 'expw':
            with open(os.path.join(data_path, self.config['label_file']), 'r') as f:
                lines = f.readlines()
            
            image_names = []
            labels = []
            for line in lines:
                parts = line.strip().split()
                image_name = parts[0]
                label = int(parts[-1])
                image_path = os.path.join(data_path, self.config['image_dir'], image_name)
                if os.path.exists(image_path):
                    image_names.append(image_name)
                    labels.append(label)
            
            self.file_names = np.array(image_names)[indices]
            self.labels = np.array(labels)[indices]
            self.file_paths = [os.path.join(data_path, self.config['image_dir'], f) 
                             for f in self.file_names]

    def __len__(self):
        if self.dataset_type == 'fer2013':
            return len(self.labels)
        else:
            return len(self.file_paths)

    def __getitem__(self, idx):
        if self.dataset_type == 'fer2013':
            pixels = self.pixels[idx].split()
            pixels = np.array([int(pixel) for pixel in pixels], dtype=np.uint8)
            image = pixels.reshape(48, 48)
            image = Image.fromarray(image)
            image = image.convert('RGB')
        else:
            path = self.file_paths[idx]
            image = Image.open(path).convert('RGB')

        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    

class DDAMFN_UnifiedDataset(data.Dataset):
    def __init__(self, dataset_type, data_path, indices, transform=None, phase='train'):
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.transform = transform
        self.phase = phase
        self.config = DDAMFN_DATASET_CONFIGS[dataset_type]

        if dataset_type == 'fer2013':
            df = pd.read_csv(os.path.join(data_path, self.config['label_file']))
            self.df = df.iloc[indices]
        elif dataset_type == 'expw':
            # Read label file and get valid images
            label_file = os.path.join(data_path, self.config['label_file'])
            image_labels = {}
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    img_name = parts[0]
                    label = int(parts[-1])
                    image_path = os.path.join(data_path, self.config['image_dir'], img_name)
                    if os.path.exists(image_path):
                        image_labels[img_name] = label
            
            # Convert to lists and apply indices
            all_images = list(image_labels.keys())
            all_labels = list(image_labels.values())
            self.file_names = np.array(all_images)[indices]
            self.labels = np.array(all_labels)[indices]
            self.file_paths = [os.path.join(data_path, self.config['image_dir'], f) 
                             for f in self.file_names]
        else:  # rafdb and ferplus
            if dataset_type == 'rafdb':
                df = pd.read_csv(os.path.join(data_path, self.config['label_file']), 
                               sep=' ', header=None, names=['name', 'label'])
            else:
                df = pd.read_csv(os.path.join(data_path, self.config['label_file']))
            
            self.file_names = df['name'].values[indices] if 'name' in df else df['Image name'].values[indices]
            self.labels = df['label'].values[indices]
            if self.config['subtract_label']:
                self.labels = self.labels - 1
            
            self.file_paths = [os.path.join(data_path, self.config['image_dir'], 
                             f.split(".")[0] + "_aligned.jpg" if dataset_type == 'rafdb' else f) 
                             for f in self.file_names]

    def __len__(self):
        if self.dataset_type == 'fer2013':
            return len(self.df)
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.dataset_type == 'fer2013':
            row = self.df.iloc[idx]
            pixels = np.array([int(pixel) for pixel in row['pixels'].split()], 
                            dtype=np.uint8)
            image = pixels.reshape((48, 48))
            image = Image.fromarray(image)
            label = row['emotion']
        else:
            image = Image.open(self.file_paths[idx]).convert('RGB')
            label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    
class AdaDF_UnifiedDataset(data.Dataset):
    def __init__(self, dataset_type, data_path, transform=None, image_dir=None):
        self.transform = transform
        self.dataset_type = dataset_type
        self.config = AdaDF_DATASET_CONFIGS[dataset_type]
        
        if self.config['is_csv']:
            if dataset_type == 'fer2013':
                # FER2013: Load pixels and labels from CSV
                df = pd.read_csv(data_path)
                self.pixels = df[self.config['pixels_col']].values
                self.labels = df[self.config['label_col']].values
            else:
                # RAFDB and FERPlus: Load image names and labels from CSV
                sep = self.config.get('csv_sep', ',')
                df = pd.read_csv(data_path, sep=sep)
                self.file_names = df[self.config['name_col']].values
                self.labels = df[self.config['label_col']].values
                if self.config.get('subtract_label', False):
                    self.labels = self.labels - 1
                
                if dataset_type == 'rafdb':
                    self.file_paths = [os.path.join(self.config['data_path'], 
                                                  self.config['image_dir'],
                                                  f.split(".")[0] + "_aligned.jpg") 
                                     for f in self.file_names]
                else:
                    self.file_paths = [os.path.join(image_dir, f) 
                                     for f in self.file_names]
        else:
            # ExpW: Load from label file
            self.image_dir = image_dir
            self.images = []
            self.labels = []
            
            with open(data_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    image_name = parts[0]
                    image_path = os.path.join(image_dir, image_name)
                    
                    if os.path.exists(image_path):
                        self.images.append(image_name)
                        self.labels.append(int(parts[-1]))
            
            self.file_paths = [os.path.join(image_dir, img) for img in self.images]
            
    def __len__(self):
        if self.dataset_type == 'fer2013':
            return len(self.labels)
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.dataset_type == 'fer2013':
            pixels = np.array([int(pixel) for pixel in self.pixels[idx].split()], dtype=np.uint8)
            image = pixels.reshape(48, 48)
            image = Image.fromarray(image)
            label = self.labels[idx]
        else:
            image = Image.open(self.file_paths[idx]).convert('RGB')
            label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx