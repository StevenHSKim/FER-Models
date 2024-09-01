'''
(표준 출력과 로그 파일에 메시지를 동시에 기록하는) 로깅 시스템 초기화
'''

import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    """
    표준 출력과 동시에 로그 파일에 메시지를 기록하는 로거를 초기화
    로그는 문자열이어야 함
    """
    LOGROOT = './log/'  # 로그 파일이 저장될 기본 경로
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  # 로그 디렉토리가 없다면 생성

    # 로그 파일 이름 생성
    logfilename = '{}-{}-{}.log'.format(config['model'], config['dataset'], get_local_time())
    logfilepath = os.path.join(LOGROOT, logfilename) # 로그 파일 경로

    # 파일 로거 설정
    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    # 스트림 로거 설정
    sfmt = u"%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    
    # 로그 레벨 설정
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    # 파일 핸들러 설정
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    # 스트림 핸들러 설정
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)
    
    # 로깅 기본 설정
    logging.basicConfig(
        level=level,
        #handlers=[sh]
        handlers = [sh, fh] # 스트림과 파일 핸들러를 로거에 추가
    )