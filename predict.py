# predict.py
import numpy as np
from config import Config
import logging

logger = logging.getLogger(__name__)

class Predictor:
    """预测器（用于新数据）"""
    
    def __init__(self, model, preprocessor=None, feature_extractor=None, config=Config):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.config = config
        
    def predict_single(self, eeg_signal: np.ndarray) -> int:
        """对单次EEG数据进行预测"""
        # 预处理
        if self.preprocessor is not None:
            # 需要实现单样本预处理
            pass
        
        # 特征提取
        if self.feature_extractor is not None:
            features = self.feature_extractor.combine_features(
                eeg_signal.reshape(1, -1)
            )
        else:
            features = eeg_signal.reshape(1, -1)
        
        # 预测
        prediction = self.model.predict(features)
        
        return prediction[0]
    
    def predict_batch(self, eeg_signals: np.ndarray) -> np.ndarray:
        """批量预测"""
        predictions = []
        for signal in eeg_signals:
            pred = self.predict_single(signal)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_with_confidence(self, eeg_signal: np.ndarray) -> tuple:
        """预测并返回置信度"""
        # 提取特征
        if self.feature_extractor is not None:
            features = self.feature_extractor.combine_features(
                eeg_signal.reshape(1, -1)
            )
        else:
            features = eeg_signal.reshape(1, -1)
        
        # 获取概率
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            prediction = np.argmax(proba)
            confidence = np.max(proba)
        else:
            prediction = self.model.predict(features)[0]
            confidence = None
        
        return prediction, confidence