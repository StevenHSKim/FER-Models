'''
실험 실행 및 크로스 밸리데이션을 관리
'''


import os
import torch
import platform
from logging import getLogger

from utils.configurator import Config
from utils.logger import init_logger
from utils.dataset import RafDataSet, shuffle_split_data
from utils.dataloader import get_datasets, TrainDataLoader, EvalDataLoader
from utils.utils import *
from common.trainer import Trainer

def run_experiment(config, train_val_indices, test_indices, iteration=0):
    """
    단일 실험 실행 함수
    """
    init_logger(config)
    logger = getLogger()
    logger.info('██Server: \t' + platform.node())
    logger.info('██Directory: \t' + os.getcwd())
    logger.info('██Configuration: \t' + str(config))

    # 데이터 설정
    train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = get_datasets(config, train_val_indices, test_indices)
    logger.info(f"██Data Setup - Train set size: {train_size}, Validation set size: {val_size}, Test set size: {test_size}")
    
    # 데이터로더 설정
    train_loader = TrainDataLoader(config, train_dataset, shuffle=True)
    val_loader = EvalDataLoader(config, val_dataset)
    test_loader = EvalDataLoader(config, test_dataset)

    # 모델 초기화
    model = get_model(config['model'], config)
    if config.get('use_parallel', False):
        model = torch.nn.DataParallel(model).to(config['device']) # 병렬 처리 활성화
        logger.info("██Using DataParallel")
    else:
        model = model.to(config['device'])
    
    logger.info(f"██Model Loaded - {config['model']}")

    # 트레이너 초기화 및 훈련 실행
    trainer = Trainer(config, model, config['device'])
    model, best_acc, best_checkpoint_path = trainer.train(train_loader, val_loader)
    logger.info("██Training Complete")
    
    # 테스트 실행
    test_acc, test_balanced_acc, test_running_loss, cm = trainer.test(test_loader, best_checkpoint_path)
    logger.info(f"██Test Results - Accuracy: {test_acc}, Balanced Accuracy: {test_balanced_acc}, Running Loss: {test_running_loss}")

    return test_acc, best_acc, test_balanced_acc, test_running_loss


def run_CrossValidation(model, dataset, config_dict):
    """
    위 단일 실험 함수를 10회 반복하는 함수
    """
    config = Config(model=model, dataset=dataset, config_dict=config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info('██Configuration Initialized for CrossValidation.')

    datapath = config.get('raf_path', '/default/path/to/dataset/')
    file_paths, labels = RafDataSet.load_raf_data(datapath)
    splits = shuffle_split_data(file_paths, labels, config)

    all_accuracies = []
    best_accuracies = []
    results = []

    # 단일 실험 반복 실행
    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        logger.info(f"██CrossValidation Iteration {iteration + 1}/{config['iterations']}")
        control_random_seed(iteration)
        test_acc, best_acc, test_balanced_acc, test_running_loss = run_experiment(config, train_val_indices, test_indices, iteration)
        all_accuracies.append(test_acc)
        best_accuracies.append(best_acc)
        results.append([iteration + 1, test_acc, test_balanced_acc, test_running_loss])
        logger.info(f"██Iteration {iteration + 1} Complete: Test Acc: {test_acc}, Best Acc: {best_acc}")

    save_results(results, 'test_results.csv')
    mean_accuracy, std_accuracy = calculate_mean_std(all_accuracies)
    logger.info(f"██Final Report - Mean Test Accuracy over {config['iterations']} iterations: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
