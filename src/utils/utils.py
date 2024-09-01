'''
유틸리티 함수
'''

import torch
import random
import numpy as np
import pandas as pd
import importlib
import datetime


def get_local_time():
    """
    현재 시간을 문자열로 반환
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    """
    모델 이름에 따라 모델 클래스를 자동으로 선택하여 반환
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    """
    Trainer 클래스를 반환
    """
    return getattr(importlib.import_module('common.trainer'), 'Trainer')


def dict2str(result_dict):
    """
    결과 딕셔너리를 문자열로 변환
    """
    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


def control_random_seed(seed):
    """
    랜덤 시드를 설정
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
def early_stopping(val_loss, best_loss, patience_counter, patience):
    """
    조기 종료 여부를 결정
    """
    if val_loss < best_loss:
        return val_loss, 0, False
    else:
        patience_counter += 1
        if patience_counter >= patience:
            return best_loss, patience_counter, True
        else:
            return best_loss, patience_counter, False
        
        
def save_results(results, file_name):
    """
    결과를 CSV 파일로 저장
    """
    results_df = pd.DataFrame(results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Test Loss'])
    results_df.to_csv(file_name, index=False)


def calculate_mean_std(accuracies):
    """
    평균과 표준 편차를 계산
    """
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    return mean_accuracy, std_accuracy
