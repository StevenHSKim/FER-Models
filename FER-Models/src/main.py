'''
메인 실행 함수
'''

import argparse
from utils.run_experiments import run_CrossValidation
# import os
# os.environ['NUMEXPR_MAX_THREADS'] = '48' # 성능 최적화를 위한 환경 변수 설정


if __name__ == '__main__':
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DAN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='RAF-DB', help='name of datasets')

    config_dict = {
        'gpu_id': 0  # GPU 설정
    }

    args, _ = parser.parse_known_args()

    # 반복실험 실행
    run_CrossValidation(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)