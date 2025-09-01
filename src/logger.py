import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    output_dir: str,
    run_name: str,
    debug: bool = False,
    rank: int = 0,
) -> logging.Logger:
    """
    设置统一的logger，支持同时输出到控制台和文件
    
    Args:
        name: logger名称
        output_dir: 输出目录
        run_name: 运行名称
        debug: 是否为调试模式
        rank: 进程rank (用于分布式训练)
    
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 格式化器
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler - 仅在debug模式或rank=0时输出
    if debug or rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件handler - 所有进程都写入各自的日志文件
    if not debug and rank == 0:  # 非debug模式下才写入文件
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{run_name}_{timestamp}.log"
        if rank > 0:
            log_filename = f"{run_name}_{timestamp}_rank{rank}.log"
        
        log_path = os.path.join(output_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 强制刷新，避免缓冲
        file_handler.flush = lambda: file_handler.stream.flush()
    
    # 设置不向上传播
    logger.propagate = False
    
    return logger


class TqdmLoggingHandler(logging.Handler):
    """
    自定义logging handler，与tqdm兼容
    避免进度条被日志打断
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except ImportError:
            # 如果没有tqdm，回退到标准输出
            msg = self.format(record)
            print(msg)
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def get_tqdm_compatible_logger(
    name: str,
    output_dir: str, 
    run_name: str,
    debug: bool = False,
    rank: int = 0
) -> logging.Logger:
    """
    获取与tqdm兼容的logger
    
    Args:
        name: logger名称
        output_dir: 输出目录
        run_name: 运行名称  
        debug: 是否为调试模式
        rank: 进程rank
        
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 格式化器
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 使用TqdmLoggingHandler替代标准StreamHandler
    if debug or rank == 0:
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        tqdm_handler.setFormatter(formatter)
        logger.addHandler(tqdm_handler)
    
    # 文件handler保持不变
    if not debug and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{run_name}_{timestamp}.log"
        if rank > 0:
            log_filename = f"{run_name}_{timestamp}_rank{rank}.log"
        
        log_path = os.path.join(output_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 设置立即刷新
        class ImmediateFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()
        
        file_handler = ImmediateFileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger


def configure_tqdm_for_file_output(use_file_output: bool = True):
    """
    配置tqdm以适配文件输出
    
    Args:
        use_file_output: 是否输出到文件
    """
    if use_file_output:
        # 禁用tqdm的动态更新，使用简单的行输出
        import tqdm
        
        # 设置tqdm默认参数
        tqdm.tqdm.monitor_interval = 0
        
        # 可选：完全禁用进度条，只显示最终结果
        # 这可以通过设置环境变量实现
        os.environ['TQDM_DISABLE'] = '1'
    else:
        # 确保tqdm正常工作
        if 'TQDM_DISABLE' in os.environ:
            del os.environ['TQDM_DISABLE']


def log_args(logger: logging.Logger, args, title: str = "Configuration"):
    """
    格式化打印参数配置
    
    Args:
        logger: logger实例
        args: 参数对象
        title: 标题
    """
    logger.info("=" * 60)
    logger.info(f" {title} ")
    logger.info("=" * 60)
    
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args
    
    max_key_len = max(len(str(k)) for k in args_dict.keys()) if args_dict else 0
    
    for key, value in sorted(args_dict.items()):
        logger.info(f"  {str(key).ljust(max_key_len)} : {value}")
    
    logger.info("=" * 60)