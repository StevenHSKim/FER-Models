import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import Poster_UnifiedDataset, DAN_UnifiedDataset, DDAMFN_UnifiedDataset, AdaDF_UnifiedDataset
from data_utils import poster_get_data_transforms, dan_get_transforms, ddamfn_get_transforms, adadf_get_transforms
from config.dataset_configs import *


#### POSTER ####

def poster_load_dataset(args):
    dataset_config = POSTER_DATASET_CONFIGS[args.dataset]
    
    if args.dataset == 'rafdb':
        df = pd.read_csv(os.path.join(dataset_config['data_path'], 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        file_names = df.iloc[:, 0].values
        labels = df.iloc[:, 1].values - 1
        file_paths = [os.path.join(dataset_config['data_path'], 'Image/aligned', f.split(".")[0] + "_aligned.jpg") for f in file_names]
        
    elif args.dataset == 'fer2013':
        df = pd.read_csv(os.path.join(dataset_config['data_path'], 'fer2013_modified.csv'))
        file_paths = np.array([np.array([int(p) for p in pixel.split()]) for pixel in df['pixels']])
        labels = df['emotion'].values
        
    elif args.dataset == 'ferplus':
        df = pd.read_csv(os.path.join(dataset_config['data_path'], 'FERPlus_Label_modified.csv'))
        file_names = df['Image name'].values
        labels = df['label'].values
        file_paths = [os.path.join(dataset_config['data_path'], 'FERPlus_Image', name) for name in file_names]
        
    elif args.dataset == 'expw':
        label_file = os.path.join(dataset_config['data_path'], 'label/label.lst')
        with open(label_file, 'r') as f:
            lines = f.readlines()
        file_names, labels = [], []
        for line in lines:
            parts = line.strip().split()
            img_name = parts[0]
            label = int(parts[-1])
            img_path = os.path.join(dataset_config['data_path'], 'aligned_image', img_name)
            if os.path.exists(img_path):
                file_names.append(img_name)
                labels.append(label)
        file_paths = [os.path.join(dataset_config['data_path'], 'aligned_image', f) for f in file_names]
        labels = np.array(labels)
    else:
        raise ValueError("Unsupported dataset!")

    return file_paths, labels, dataset_config


def poster_get_datasets(file_paths, labels, train_indices, val_indices, test_indices, args):
    train_dataset = Poster_UnifiedDataset(
        np.array(file_paths)[train_indices],
        np.array(labels)[train_indices],
        transform=poster_get_data_transforms(train=True),
        basic_aug=True,
        dataset_type=args.dataset
    )

    val_dataset = Poster_UnifiedDataset(
        np.array(file_paths)[val_indices],
        np.array(labels)[val_indices],
        transform=poster_get_data_transforms(train=False),
        dataset_type=args.dataset
    )

    test_dataset = Poster_UnifiedDataset(
        np.array(file_paths)[test_indices],
        np.array(labels)[test_indices],
        transform=poster_get_data_transforms(train=False),
        dataset_type=args.dataset
    )

    return train_dataset, val_dataset, test_dataset, len(train_dataset), len(val_dataset), len(test_dataset)


def poster_get_dataloaders(train_dataset, val_dataset, test_dataset, args):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader



#### DAN ####

def dan_load_dataset(args):
    dataset_config = DAN_DATASET_CONFIGS[args.dataset]
    
    # Load data based on dataset type
    if args.dataset == 'fer2013':
        df = pd.read_csv(os.path.join(dataset_config['data_path'], dataset_config['label_file']))
        indices = np.arange(len(df))
        labels = df['emotion'].values
    elif args.dataset == 'expw':
        image_names = []
        labels = []
        with open(os.path.join(dataset_config['data_path'], dataset_config['label_file']), 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name = parts[0]
                label = int(parts[-1])
                image_path = os.path.join(dataset_config['data_path'], dataset_config['image_dir'], image_name)
                if os.path.exists(image_path):
                    image_names.append(image_name)
                    labels.append(label)
        indices = np.arange(len(image_names))
        labels = np.array(labels)
    else:  # rafdb and ferplus
        df = pd.read_csv(os.path.join(dataset_config['data_path'], dataset_config['label_file']),
                        sep=' ' if args.dataset == 'rafdb' else ',',
                        header=None if args.dataset == 'rafdb' else 0)
        indices = np.arange(len(df))
        if args.dataset == 'rafdb':
            labels = df[1].values - 1
        else:
            labels = df['label'].values
            if labels.min() > 0:
                labels = labels - 1
    return indices, labels

def dan_get_dataloaders(train_indices, val_indices, test_indices, args):
    """Create and return DataLoaders for training, validation, and testing."""
    dataset_config = DAN_DATASET_CONFIGS[args.dataset]
    
    # Create datasets
    train_dataset = DAN_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        train_indices,
        transform=dan_get_transforms(train=True),
        phase='train'
    )

    val_dataset = DAN_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        val_indices,
        transform=dan_get_transforms(train=False),
        phase='validation'
    )

    test_dataset = DAN_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        test_indices,
        transform=dan_get_transforms(train=False),
        phase='test'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader



#### DDAMFN ####

def ddamfn_load_datasets(args):
    dataset_config = DDAMFN_DATASET_CONFIGS[args.dataset]
    
    # Load data based on dataset type
    if args.dataset == 'fer2013':
        df = pd.read_csv(os.path.join(dataset_config['data_path'], dataset_config['label_file']))
        indices = np.arange(len(df))
        labels = df['emotion'].values
        
    elif args.dataset == 'expw':
        label_file = os.path.join(dataset_config['data_path'], dataset_config['label_file'])
        image_labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                label = int(parts[-1])
                img_path = os.path.join(dataset_config['data_path'], 
                                      dataset_config['image_dir'], img_name)
                if os.path.exists(img_path):
                    image_labels[img_name] = label
        indices = np.arange(len(image_labels))
        labels = np.array(list(image_labels.values()))
    
    else:  # rafdb and ferplus
        if args.dataset == 'rafdb':
            df = pd.read_csv(os.path.join(dataset_config['data_path'], dataset_config['label_file']), sep=' ', header=None, names=['name', 'label'])
        else:
            df = pd.read_csv(os.path.join(dataset_config['data_path'], dataset_config['label_file']))
        indices = np.arange(len(df))
        labels = df['label'].values
        if dataset_config['subtract_label']:
            labels = labels - 1
    
    return indices, labels

def ddamfn_get_dataloaders(train_indices, val_indices, test_indices, args):
    dataset_config = DDAMFN_DATASET_CONFIGS[args.dataset]
    
    # Create datasets
    train_dataset = DDAMFN_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        train_indices,
        transform=ddamfn_get_transforms(train=True),
        phase='train'
    )

    val_dataset = DDAMFN_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        val_indices,
        transform=ddamfn_get_transforms(train=False),
        phase='validation'
    )

    test_dataset = DDAMFN_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        test_indices,
        transform=ddamfn_get_transforms(train=False),
        phase='test'
    )

    print(f'Train set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')
    print(f'Test set size: {len(test_dataset)}')

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


#### AdaDF ####

def adadf_load_datasets(args):
    """Load dataset indices and labels similar to DDAMFN structure"""
    dataset_config = AdaDF_DATASET_CONFIGS[args.dataset]
    
    if dataset_config['is_csv']:
        if args.dataset == 'fer2013':
            df = pd.read_csv(dataset_config['data_path'])
            indices = np.arange(len(df))
            labels = df[dataset_config['label_col']].values
        else:  # RAFDB & FERPlus
            sep = dataset_config.get('csv_sep', ',')
            df = pd.read_csv(dataset_config['data_path'], sep=sep)
            indices = np.arange(len(df))
            labels = df[dataset_config['label_col']].values
            if dataset_config.get('subtract_label', False):
                labels = labels - 1
    else:  # ExpW
        dataset = AdaDF_UnifiedDataset(
            args.dataset, 
            dataset_config['data_path'], 
            image_dir=dataset_config['image_dir']
        )
        indices = np.arange(len(dataset))
        labels = np.array(dataset.labels)
        
    return indices, labels

def adadf_get_dataloaders(train_indices, val_indices, test_indices, args):
    """Create dataloaders similar to DDAMFN structure"""
    dataset_config = AdaDF_DATASET_CONFIGS[args.dataset]
    image_dir = dataset_config.get('image_dir', None)
    
    # Create base datasets
    train_dataset = AdaDF_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        transform=adadf_get_transforms(train=True),
        image_dir=image_dir
    )
    
    val_dataset = AdaDF_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        transform=adadf_get_transforms(train=False),
        image_dir=image_dir
    )
    
    test_dataset = AdaDF_UnifiedDataset(
        args.dataset,
        dataset_config['data_path'],
        transform=adadf_get_transforms(train=False),
        image_dir=image_dir
    )
    
    # Create subsets using indices
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Print dataset sizes
    print(f'Train set size: {len(train_subset)}')
    print(f'Validation set size: {len(val_subset)}')
    print(f'Test set size: {len(test_subset)}')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader