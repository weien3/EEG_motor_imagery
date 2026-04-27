# config.py
import os
from pathlib import Path

class Config:
    # ========== 数据路径配置 ==========
    # 项目根目录
    PROJECT_ROOT = Path(r"C:\BCI\分析实训项目\eeg_motor_imagery")
    
    # 数据集根目录（根据你的实际路径）
    DATA_ROOT = Path(r"C:\BCI\分析实训项目\eeg-motor-movementimagery-dataset-1.0.0\files")
    
    # 缓存目录
    CACHE_DIR = PROJECT_ROOT / "cache"
    
    # 输出目录
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODEL_SAVE_PATH = OUTPUT_DIR / "models"
    RESULT_PATH = OUTPUT_DIR / "results"
    LOG_PATH = PROJECT_ROOT / "logs"
    
    # ========== 数据选择配置 ==========
    # 要处理的被试列表（S001, S002, ...）
    SUBJECTS = [f"S{i:03d}" for i in range(1, 110)]  # S001 到 S109
    
    # 或者只测试前几个被试
    # SUBJECTS = ["S001", "S002", "S003"]
    
    # 要使用的run（R01, R02, ...）
    # 通常：R01=睁眼基线，R02=闭眼基线，R03-R14=任务run
    TASK_RUNS = [f"R{i:02d}" for i in range(3, 15)]  # R03 到 R14
    BASELINE_RUNS = ["R01", "R02"]  # 基线run
    
    # ========== 预处理参数 ==========
    LOW_FREQ = 0.5      # 高通滤波下限(Hz)
    HIGH_FREQ = 40.0    # 低通滤波上限(Hz)
    NOTCH_FREQ = 50.0   # 陷波频率(Hz)
    
    # 提取epoch的时间窗口（秒）
    TMIN = 0.5          # 提示出现后开始时间
    TMAX = 4.0          # 结束时间
    
    # 采样频率（数据集是160Hz）
    SFREQ = 160
    
    # ========== 特征提取参数 ==========
    CSP_N_COMPONENTS = 6
    CSP_LOG = True
    
    # 频带
    FREQ_BANDS = {
        'mu': [8, 13],
        'beta': [13, 30],
        'alpha': [8, 12],
        'low_beta': [12, 20],
        'high_beta': [20, 30]
    }
    
    # ========== 数据划分 ==========
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # ========== 模型参数 ==========
    MODEL_TYPE = 'cnn'  # 'svm', 'rf', 'cnn'
    
    # SVM参数
    SVM_C = 1.0
    SVM_GAMMA = 'scale'
    SVM_KERNEL = 'rbf'
    
    # 随机森林参数
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    
    # 类别权重
    USE_CLASS_WEIGHTS = True
    
    # ========== 深度学习参数 ==========
    BATCH_SIZE = 32
    N_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # ========== 其他配置 ==========
    VERBOSE = True
    N_JOBS = -1  # 使用所有CPU核心
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        cls.RESULT_PATH.mkdir(parents=True, exist_ok=True)
        cls.LOG_PATH.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_subject_path(cls, subject_id):
        """获取被试数据文件夹路径"""
        return cls.DATA_ROOT / subject_id
    
    @classmethod
    def get_subject_files(cls, subject_id, runs=None):
        """获取被试的指定run文件列表"""
        subject_path = cls.get_subject_path(subject_id)
        if not subject_path.exists():
            return []
        
        if runs is None:
            runs = cls.TASK_RUNS
        
        files = []
        for run in runs:
            # 文件命名格式：S001R01.edf
            file_pattern = f"{subject_id}{run}.edf"
            file_path = subject_path / file_pattern
            if file_path.exists():
                files.append(file_path)
        
        return files
    
    @classmethod
    def print_info(cls):
        """打印配置信息"""
        print("=" * 50)
        print("Configuration")
        print("=" * 50)
        print(f"Data root: {cls.DATA_ROOT}")
        print(f"Subjects: {cls.SUBJECTS[:5]}... (total {len(cls.SUBJECTS)})")
        print(f"Task runs: {cls.TASK_RUNS}")
        print(f"Output dir: {cls.OUTPUT_DIR}")
        print(f"Model type: {cls.MODEL_TYPE}")
        print("=" * 50)