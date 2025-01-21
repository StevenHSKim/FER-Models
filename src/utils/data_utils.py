import numpy as np
import cv2
from torchvision import transforms
from sklearn.model_selection import ShuffleSplit
from models.adadf.auto_augment import rand_augment_transform # AdaDF



### Overall ###
        
def shufflesplit_datasets(file_paths, labels, args):
    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(file_paths, labels))
    return splits



#### POSTER ####

def poster_add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def poster_flip_image(image_array):
    return cv2.flip(image_array, 1)

def poster_get_data_transforms(train=True):
    if train:
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

        
#### DAN ####
        
def dan_get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        

### DDAMFN ###

def ddamfn_get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(5), transforms.RandomCrop(112, padding=8)], p=0.2),
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
        
        
        
### AdaDF ###

def adadf_get_transforms(train=True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            rand_augment_transform('rand-m5-n3-mstd0.5', {'translate_const': 117, 'img_mean': (124, 116, 104)}),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])