# quick_evaluate.py
import torch
import numpy as np
from pathlib import Path
from config import Config
from data_loader import EEGDataLoader
from preprocess import EEGPreprocessor
from evaluate import Evaluator
import joblib

# 加载数据
loader = EEGDataLoader()
X, y, subject_info = loader.load_multiple_subjects(max_subjects=3)

# 预处理
preprocessor = EEGPreprocessor()
X_normalized = np.zeros_like(X)
for i in range(X.shape[0]):
    X_normalized[i] = preprocessor.normalize_data(X[i], method='zscore')

# 划分测试集（使用最后20%作为测试）
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    X_normalized, y, test_size=0.2, stratify=y, random_state=42
)

# 准备CNN数据
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)

# 加载模型
from train import ModelTrainer
trainer = ModelTrainer()

# 根据模型类型加载
if Config.MODEL_TYPE == 'cnn':
    model_path = Path(Config.MODEL_SAVE_PATH) / "cnn_model.pkl"
    # 需要手动创建模型实例
    from models.eegnet import EEGNet
    model = EEGNet(
        n_channels=X_test.shape[1],
        n_classes=len(np.unique(y)),
        n_timepoints=X_test.shape[2]
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
else:
    model_path = Path(Config.MODEL_SAVE_PATH) / f"{Config.MODEL_TYPE}_model.pkl"
    model = joblib.load(model_path)

# 评估
evaluator = Evaluator()
results = evaluator.evaluate(model, X_test_cnn if Config.MODEL_TYPE == 'cnn' else X_test, y_test)
evaluator.plot_confusion_matrix(save_path="confusion_matrix.png")