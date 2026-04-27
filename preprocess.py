# preprocess.py
import numpy as np
import mne
from scipy import signal
from config import Config
from typing import Tuple, Optional

class EEGPreprocessor:
    """EEG数据预处理器"""
    
    def __init__(self, config=Config):
        self.config = config
        self.ica = None
        
    def filter_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """应用带通滤波和陷波滤波"""
        # 带通滤波
        raw.filter(
            self.config.LOW_FREQ, 
            self.config.HIGH_FREQ,
            fir_design='firwin',
            verbose=False
        )
        
        # 陷波滤波（去除工频干扰）
        if self.config.NOTCH_FREQ:
            raw.notch_filter(
                self.config.NOTCH_FREQ,
                fir_design='firwin',
                verbose=False
            )
        
        return raw
    
    def extract_epochs(self, raw: mne.io.Raw) -> Tuple[np.ndarray, np.ndarray]:
        """提取任务相关的epochs"""
        # 从注释中提取事件
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # 创建epochs
        epochs = mne.Epochs(
            raw, 
            events, 
            event_id, 
            tmin=self.config.TMIN,
            tmax=self.config.TMAX,
            baseline=None,
            preload=True,
            verbose=False
        )
        
        # 提取数据
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        y = epochs.events[:, -1]  # 标签
        
        # 重新映射标签（如果需要）
        y = self._remap_labels(y, event_id)
        
        return X, y
    
    def _remap_labels(self, y: np.ndarray, event_id: dict) -> np.ndarray:
        """将事件ID重新映射为连续的类别标签"""
        # 获取所有唯一的事件ID
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[label] for label in y])
    
    def apply_ica(self, epochs: mne.Epochs, n_components: int = 20) -> mne.Epochs:
        """应用ICA去除伪迹"""
        from mne.preprocessing import ICA
        
        # 创建ICA对象
        self.ica = ICA(
            n_components=n_components,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        
        # 拟合ICA
        self.ica.fit(epochs, verbose=False)
        
        # 自动检测并排除眼电成分（简化版）
        # 实际使用时需要更精细的判断
        # self.ica.exclude = self._detect_eog_components(epochs)
        
        # 应用ICA
        epochs_clean = self.ica.apply(epochs, verbose=False)
        
        return epochs_clean
    
    def extract_baseline(self, raw: mne.io.Raw, duration: float = 60.0) -> np.ndarray:
        """提取基线数据（睁眼/闭眼静息态）"""
        # 找到基线时间段
        # 实际实现需要根据数据集的标注来确定
        return np.array([])
    
    def normalize_data(self, X: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """数据归一化"""
        if method == 'zscore':
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
            X_normalized = (X - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            min_val = X.min(axis=-1, keepdims=True)
            max_val = X.max(axis=-1, keepdims=True)
            X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
        
        else:
            X_normalized = X
        
        return X_normalized