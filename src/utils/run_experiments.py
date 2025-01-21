import os
import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
# 기존 train/test 함수 import 제거
from common.trainer import get_trainer, train  # 새로운 trainer 모듈 import
from tqdm import tqdm

# Import common utilities
from utils.utils import (
    control_random_seed, 
    calculate_model_parameters,
    calculate_metrics,
    save_iteration_results,
    update_total_results,
    print_final_results
)

# Import data utilities
from utils.data_utils import shufflesplit_datasets

# Import dataset configurations
from config.dataset_configs import (
    POSTER_DATASET_CONFIGS,
    DAN_DATASET_CONFIGS,
    DDAMFN_DATASET_CONFIGS,
    AdaDF_DATASET_CONFIGS
)

# Import dataloaders
from utils.dataloader import (
    poster_load_dataset,
    poster_get_datasets,
    poster_get_dataloaders,
    dan_load_dataset,
    dan_get_dataloaders,
    ddamfn_load_datasets,
    ddamfn_get_dataloaders,
    adadf_load_datasets,
    adadf_get_dataloaders
)

# Import models
from models.poster.poster import pyramid_trans_expr
from models.dan.dan import DAN
from models.ddamfn import DDAMNet
from models.adadf.resnet18 import create_model

# Import training functions from common trainer
from common.trainer import (
    train_poster,
    test_poster,
    train_dan,
    test_dan,
    train_ddamfn,
    test_ddamfn,
    train_adadf,
    test_adadf
)

# Import losses
from common.loss import (
    POSTER_LabelSmoothingCrossEntropy,
    DAN_AffinityLoss,
    DAN_PartitionLoss,
    DDAMFN_AttentionLoss
)

def setup_model_specific_components(args):
    """Setup model-specific configurations and components"""
    
    if args.model_name == "POSTER":
        dataset_config = POSTER_DATASET_CONFIGS[args.dataset]
        model = pyramid_trans_expr(img_size=224, num_classes=dataset_config['num_classes'], type=args.modeltype)
        model = torch.nn.DataParallel(model).cuda()
        criterion = {
            'CE': torch.nn.CrossEntropyLoss(),
            'lsce': POSTER_LabelSmoothingCrossEntropy(smoothing=0.2)
        }
        
    elif args.model_name == "DAN":
        dataset_config = DAN_DATASET_CONFIGS[args.dataset]
        model = DAN(num_head=args.num_head, num_class=dataset_config['num_classes']).cuda()
        criterion = {
            'cls': torch.nn.CrossEntropyLoss(),
            'af': DAN_AffinityLoss(device='cuda', num_class=dataset_config['num_classes']),
            'pt': DAN_PartitionLoss()
        }
        
    elif args.model_name == "DDAMFN":
        dataset_config = DDAMFN_DATASET_CONFIGS[args.dataset]
        model = DDAMNet(num_class=dataset_config['num_classes'], num_head=args.num_head)
        model = torch.nn.DataParallel(model).cuda()
        criterion = {
            'cls': torch.nn.CrossEntropyLoss(),
            'at': DDAMFN_AttentionLoss()
        }
        
    elif args.model_name == "AdaDF":
        dataset_config = AdaDF_DATASET_CONFIGS[args.dataset]
        model = create_model(dataset_config['num_classes'], args.drop_rate).cuda()
        criterion = {
            'cls': torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing),
            'kld': torch.nn.KLDivLoss(reduction='none')
        }
        
    return model, criterion, dataset_config

def setup_optimizer(model_params, optimizer_type, lr, weight_decay):
    """Setup optimizer based on type"""
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def setup_scheduler(optimizer, scheduler_type, t_max):
    """Setup learning rate scheduler based on type"""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def run_train_test(args):
    """Main function to run training and testing experiments"""
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")
    
    # Initialize tracking lists for metrics
    all_accuracies = []
    all_balanced_accuracies = []
    best_accuracies = []
    all_val_losses = []
    all_test_losses = []
    results = []
    
    # Setup model-specific components
    model, criterion, dataset_config = setup_model_specific_components(args)
    
    # Calculate and print model parameters
    parameters = calculate_model_parameters(model)
    print(f'Total Parameters: {parameters:.3f}M')
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(
        model_params=model.parameters(),
        optimizer_type=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = setup_scheduler(
        optimizer=optimizer,
        scheduler_type=args.lr_scheduler,
        t_max=args.t_max
    )
    
    # Initialize trainer
    trainer = get_trainer(
        model_name=args.model_name,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args
    )
    
    # Load dataset based on model type
    if args.model_name == "POSTER":
        file_paths, labels, dataset_config = poster_load_dataset(args)
        indices = file_paths  # For consistency with other models
    elif args.model_name == "DAN":
        indices, labels = dan_load_dataset(args)
    elif args.model_name == "DDAMFN":
        indices, labels = ddamfn_load_datasets(args)
    else:  # AdaDF
        indices, labels = adadf_load_datasets(args)
    
    # Create splits for repeated experiments
    splits = shufflesplit_datasets(indices, labels, args)
    
    # Run iterations
    for iteration, (train_val_indices, test_indices) in enumerate(splits, 1):
        print(f"\nIteration {iteration}/{args.iterations}")
        
        # Set random seed for reproducibility
        control_random_seed(iteration)
        
        # Split train and validation sets
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=args.val_size,
            random_state=iteration
        )
        
        print(f'Train set size: {len(train_indices)}')
        print(f'Validation set size: {len(val_indices)}')
        print(f'Test set size: {len(test_indices)}')
        
        # Create dataloaders based on model type
        if args.model_name == "POSTER":
            train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = poster_get_datasets(
                file_paths, labels, train_indices, val_indices, test_indices, args
            )
            train_loader, val_loader, test_loader = poster_get_dataloaders(
                train_dataset, val_dataset, test_dataset, args
            )
        elif args.model_name == "DAN":
            train_loader, val_loader, test_loader = dan_get_dataloaders(
                args.dataset, train_indices, val_indices, test_indices, args, iteration
            )
        elif args.model_name == "DDAMFN":
            train_loader, val_loader, test_loader = ddamfn_get_dataloaders(
                train_indices, val_indices, test_indices, args
            )
        else:  # AdaDF
            train_loader, val_loader, test_loader = adadf_get_dataloaders(
                train_indices, val_indices, test_indices, args
            )
        
        # Train the model using trainer
        best_checkpoint_path, best_acc, best_val_loss = train(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            iteration=iteration
        )
        
        # Test the model if training was successful
        if best_checkpoint_path:
            test_acc, test_balanced_acc, test_loss = trainer.test(
                test_loader=test_loader,
                checkpoint_path=best_checkpoint_path
            )
            
            # Record results
            all_accuracies.append(test_acc)
            all_balanced_accuracies.append(test_balanced_acc)
            best_accuracies.append(best_acc)
            all_val_losses.append(best_val_loss)
            all_test_losses.append(test_loss)
            results.append([iteration, test_acc, test_balanced_acc, best_val_loss, test_loss])
            
            print(f"\nIteration {iteration} Results:")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test Balanced Accuracy: {test_balanced_acc:.4f}")
            print(f"Best Validation Accuracy: {best_acc:.4f}")
            print(f"Final Test Loss: {test_loss:.4f}")
    
    # Get current time for experiment logging
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    
    # Calculate final metrics
    metrics = calculate_metrics(
        accuracies=all_accuracies,
        balanced_accuracies=all_balanced_accuracies,
        val_losses=all_val_losses,
        test_losses=all_test_losses
    )
    
    # Save results
    save_iteration_results(args, metrics, results)
    update_total_results(args, metrics, current_time, parameters)
    
    # Print final results
    print_final_results(args, metrics, best_accuracies)
    
    return metrics, results