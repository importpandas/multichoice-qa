#coding=utf8
""" Utility functions include:
1. set output logging path
2. set random seed for all libs
3. select torch.device
"""
import sys, os, logging
import random, torch
import numpy as np

logger = logging.getLogger(__name__)

def setup_root_logger(checkpoint_dir, rank, debug, postfix=""):
    """Setup the root logger for all the packages

    log lines with WARNING or higher level will be printed
    to stderr, while stderr will be collected as ERROR level
    """
    # log file prepare
    log_dir = checkpoint_dir / ('log' + postfix)
    try:
        log_dir.mkdir(mode=0o755)
    except FileExistsError:
        pass
    log_file = log_dir / 'rank{}.log'.format(rank)

    basic_handler = logging.FileHandler(log_file)
    basic_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handlers = [basic_handler]

    stdout_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.setLevel(logging.WARNING)
    handlers.append(stdout_handler)

    if rank == 0:
        critical_handler = logging.FileHandler(log_dir / 'critical.log')
        critical_handler.setLevel(logging.WARNING)
        handlers.append(critical_handler)

    #package_name = __name__.split('.')[0]
    #root_logger = logging.getLogger(package_name)
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        '%(asctime)s[%(name)s]-%(levelname)s-%(message)s',
        datefmt='%H:%M:%S',
    )
    root_logger.setLevel(logging.DEBUG)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Capture python's builtin warnings message
    logging.captureWarnings(True)
    redirect_stderr()


def redirect_stderr():
    """Redirect stderr to logger"""

    class LoggerWriter:
        """https://github.com/apache/airflow/pull/6767/files"""
        def __init__(self, target_logger, level=logging.INFO):
            self.logger = target_logger
            self.level = level

        def write(self, message):
            if message and not message.isspace():
                self.logger.log(self.level, message)

        def fileno(self):
            """
            Returns the stdout file descriptor 1.
            For compatibility reasons e.g python subprocess module stdout redirection.
            """
            return 1

        def flush(self):
            """MUST define flush method to exit gracefully"""

    sys.stderr = LoggerWriter(logger, logging.ERROR)


def set_logger(exp_path, testing=False, is_main_process=True):
    logFormatter = logging.Formatter('%(asctime)s - %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG if is_main_process else logging.WARNING)
    if testing:
        fileHandler = logging.FileHandler(f'{exp_path}/log_test.txt', mode='w')
    else:
        fileHandler = logging.FileHandler(f'{exp_path}/log_train.txt', mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger

def set_random_seed(random_seed=999):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

def set_torch_device(deviceId):
    if deviceId < 0:
        device = torch.device("cpu")
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device("cuda:%d" % (deviceId))
        torch.backends.cudnn.enabled = False
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # used when debug
        ## These two sentences are used to ensure reproducibility with cudnn backend
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return device