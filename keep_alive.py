import signal
import time
import threading
import psutil
import os
import torch
import logging
from datetime import datetime
import subprocess
import sys

class EnhancedKeepAlive:
    def __init__(self, interval=30, memory_threshold=0.9, auto_resume=True):
        self.interval = interval
        self.memory_threshold = memory_threshold
        self.auto_resume = auto_resume
        self._stop_event = threading.Event()
        self.last_activity = time.time()
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        self.process_priority_set = False
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_monitor.log'),
                logging.StreamHandler()
            ]
        )
    
    def update_activity(self):
        self.last_activity = time.time()
        
    def save_emergency_checkpoint(self, model, optimizer, epoch, filename=None):
        if filename is None:
            filename = f"emergency_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'last_activity': self.last_activity
        }, checkpoint_path)
        self.logger.info(f"Emergency checkpoint saved: {checkpoint_path}")
        return checkpoint_path
        
    def set_process_priority(self):
        """设置进程优先级"""
        try:
            # 获取当前进程
            process = psutil.Process(os.getpid())
            
            # 设置较高的进程优先级
            if sys.platform == 'linux':
                # Linux下设置nice值(-20到19，越小优先级越高)
                os.nice(-10)  
                # 设置I/O优先级
                subprocess.run(['ionice', '-c', '2', '-n', '0', '-p', str(os.getpid())])
            
            # 设置内存锁定，防止被换出
            if hasattr(os, 'mlockall'):
                os.mlockall(os.MCL_CURRENT | os.MCL_FUTURE)
                
            self.process_priority_set = True
            self.logger.info("Process priority settings applied successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to set process priority: {e}")

    def monitor_training(self, model, optimizer, current_epoch):
        if not self.process_priority_set:
            self.set_process_priority()
            
        while not self._stop_event.is_set():
            try:
                # 检查系统资源
                memory = psutil.virtual_memory()
                
                # 如果内存使用率过高，主动释放缓存
                if memory.percent > self.memory_threshold * 100:
                    self.logger.warning("High memory usage detected, taking preventive actions...")
                    torch.cuda.empty_cache()
                    if sys.platform == 'linux':
                        subprocess.run(['sync'])  # 同步文件系统缓存
                        subprocess.run(['echo', '3', '>', '/proc/sys/vm/drop_caches'], shell=True)
                
                # 定期保存检查点，而不是等到检测到无活动
                if time.time() - self.last_activity > self.interval:
                    self.save_emergency_checkpoint(
                        model, optimizer, current_epoch,
                        f"periodic_checkpoint_epoch_{current_epoch}.pth"
                    )
                    self.update_activity()
                
                # 监控GPU使用情况
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory_used = torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i)
                        if gpu_memory_used > self.memory_threshold:
                            self.logger.warning(f"GPU {i} memory usage high, clearing cache...")
                            torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                # 发生错误时也保存检查点
                self.save_emergency_checkpoint(
                    model, optimizer, current_epoch,
                    f"error_checkpoint_epoch_{current_epoch}.pth"
                )
            
            time.sleep(self.interval)
    
    def start(self, model=None, optimizer=None, current_epoch=None):
        # 在启动监控之前设置进程优先级
        self.set_process_priority()
        self.keep_alive_thread = threading.Thread(
            target=self.monitor_training,
            args=(model, optimizer, current_epoch),
            daemon=True
        )
        self.keep_alive_thread.start()
        self.logger.info("Training monitor started")
        
    def stop(self):
        self._stop_event.set()
        self.logger.info("Training monitor stopped") 