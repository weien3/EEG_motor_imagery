# feature_extraction.py
import numpy as np
from scipy import signal
from mne.decoding import CSP
from sklearn.decomposition import PCA
from config import Config

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, config=Config):
        self.config = config
        self.csp = None
        self.pca = None
        
    def extract_csp_features(self, X: np.ndarray, y: np.ndarray = None, 
                            fit: bool = True) -> np.ndarray:
        """提取CSP特征"""
        if fit:
            # 训练CSP
            self.csp = CSP(
                n_components=self.config.CSP_N_COMPONENTS,
                log=self.config.CSP_LOG,
                reg=None,
                norm_trace=False
            )
            features = self.csp.fit_transform(X, y)
        else:
            # 使用已训练的CSP
            if self.csp is None:
                raise ValueError("CSP not fitted yet. Call fit first.")
            features = self.csp.transform(X)
        
        return features
    
    def extract_band_power(self, X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
        """提取频带功率特征"""
        n_samples, n_channels, n_times = X.shape
        features = []
        
        for sample in X:
            sample_features = []
            for ch in range(n_channels):
                for band_name, band_range in self.config.FREQ_BANDS.items():
                    # 计算功率谱密度
                    freqs, psd = signal.welch(
                        sample[ch], 
                        fs=sfreq,
                        nperseg=min(256, n_times),
                        noverlap=128
                    )
                    
                    # 提取特定频带的平均功率
                    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
                    if np.any(mask):
                        band_power = np.mean(psd[mask])
                        sample_features.append(band_power)
                    else:
                        sample_features.append(0)
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """提取统计特征（均值、方差、偏度、峰度等）"""
        features = []
        
        for sample in X:
            sample_features = []
            for ch in range(sample.shape[0]):
                # 时域统计特征
                sample_features.extend([
                    np.mean(sample[ch]),
                    np.std(sample[ch]),
                    np.max(sample[ch]),
                    np.min(sample[ch]),
                    np.ptp(sample[ch]),  # 峰峰值
                    np.sum(sample[ch]**2) / len(sample[ch]),  # 能量
                ])
            
            # 频域特征也可加入
            features.append(sample_features)
        
        return np.array(features)
    
    def apply_pca(self, features: np.ndarray, n_components: int = 20, 
                  fit: bool = True) -> np.ndarray:
        """PCA降维"""
        if fit:
            self.pca = PCA(n_components=n_components, 
                          random_state=self.config.RANDOM_STATE)
            features_pca = self.pca.fit_transform(features)
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted yet.")
            features_pca = self.pca.transform(features)
        
        return features_pca
    
    def combine_features(self, X: np.ndarray, y: np.ndarray = None, 
                        use_csp: bool = True, 
                        use_band_power: bool = True,
                        use_stats: bool = False) -> np.ndarray:
        """组合多种特征"""
        all_features = []
        
        if use_csp and y is not None:
            csp_feat = self.extract_csp_features(X, y, fit=True)
            all_features.append(csp_feat)
            print(f"CSP features shape: {csp_feat.shape}")
        
        if use_band_power:
            band_feat = self.extract_band_power(X)
            all_features.append(band_feat)
            print(f"Band power features shape: {band_feat.shape}")
        
        if use_stats:
            stat_feat = self.extract_statistical_features(X)
            all_features.append(stat_feat)
            print(f"Statistical features shape: {stat_feat.shape}")
        
        if all_features:
            combined = np.hstack(all_features)
            return combined
        else:
            # 如果没有提取任何特征，返回原始数据（需要reshape）
            return X.reshape(X.shape[0], -1)