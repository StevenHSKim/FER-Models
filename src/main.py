import os
import sys
# 프로젝트 루트 디렉토리를 파이썬 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
from  utils.run_experiments import run_train_test

def parse_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--model_name', type=str, required=True, choices=['POSTER', 'DAN', 'DDAMFN', 'AdaDF'], help='Name of the model to run')
    parser.add_argument('--dataset', type=str, required=True, choices=['rafdb', 'fer2013', 'ferplus', 'expw'], help='Dataset to use')
    
    # Common arguments
    # GPU 관련 인자 수정
    parser.add_argument('--gpu', type=str, default='0', help='GPU device IDs to use.' 
                                                             'POSTER, DDAMFN: (e.g., "0,1")'
                                                             'DAN, AdaDF:(e.g., "0")')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--iterations', type=int, default=10, help='Number of experiment iterations')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of training data for validation')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Patience for early stopping')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # Learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'exp'], help='Learning rate scheduler type')
    parser.add_argument('--t_max', type=int, default=10, help='T_max for CosineAnnealingLR scheduler')
    
    # Model specific arguments
    # POSTER
    parser.add_argument('--modeltype', type=str, default='large', choices=['small', 'base', 'large'], help='POSTER model type')
    
    # DAN & DDAMFN
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention heads for DAN/DDAMFN')
    
    # AdaDF
    parser.add_argument('--threshold', type=float, default=0.7, help='Threshold value for AdaDF')
    parser.add_argument('--sharpen', action='store_true', help='Use sharpening in AdaDF')
    parser.add_argument('--T', type=float, default=1.2, help='Temperature parameter for AdaDF')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha parameter for AdaDF')
    parser.add_argument('--beta', type=int, default=3, help='Beta parameter for AdaDF')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Dropout rate for AdaDF')
    parser.add_argument('--max_weight', type=float, default=1.0, help='Maximum weight value for AdaDF')
    parser.add_argument('--min_weight', type=float, default=0.2, help='Minimum weight value for AdaDF')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing value for AdaDF')
    
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    print(f"\nStarting experiment with {args.model_name} model on {args.dataset} dataset")
    print(f"GPU: {args.gpu}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Iterations: {args.iterations}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr}")
    print("=" * 50)
    
    # Run experiment
    try:
        metrics, results = run_train_test(args)
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()