# train.py
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config=Config):
        self.config = config
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
    def compute_class_weights(self, y_train: np.ndarray) -> dict:
        """计算类别权重"""
        class_counts = np.bincount(y_train)
        base_weights = 1.0 / np.sqrt(class_counts + 1e-6)
        base_weights = base_weights / base_weights.mean()
        
        # 转换为字典格式
        class_weights = {i: base_weights[i] for i in range(len(class_counts))}
        
        logger.info(f"Class counts: {class_counts}")
        logger.info(f"Class weights: {class_weights}")
        
        return class_weights
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """训练SVM模型"""
        # 获取类别权重
        class_weights = None
        if self.config.USE_CLASS_WEIGHTS:
            class_weights = self.compute_class_weights(y_train)
        
        # 创建SVM模型
        self.model = SVC(
            C=self.config.SVM_C,
            kernel=self.config.SVM_KERNEL,
            gamma=self.config.SVM_GAMMA,
            class_weight=class_weights,
            probability=True,
            random_state=self.config.RANDOM_STATE
        )
        
        # 训练
        logger.info("Training SVM model...")
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """训练随机森林模型"""
        class_weights = None
        if self.config.USE_CLASS_WEIGHTS:
            class_weights = self.compute_class_weights(y_train)
        
        self.model = RandomForestClassifier(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            class_weight=class_weights,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def train_cnn(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
        """训练CNN模型"""
        import torch
        from models.eegnet import EEGNet  # 需要在models目录下实现
        
        # 准备数据
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # 添加通道维度
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
        y_val_tensor = torch.LongTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE)
        
        # 创建模型
        n_channels = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        self.model = EEGNet(n_channels=n_channels, n_classes=n_classes)
        
        # 损失函数（带类别权重）
        if self.config.USE_CLASS_WEIGHTS:
            class_weights = self.compute_class_weights(y_train)
            class_weights_tensor = torch.FloatTensor([class_weights[i] for i in range(n_classes)])
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        # 训练循环
        for epoch in range(self.config.N_EPOCHS):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / len(val_dataset)
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.N_EPOCHS}, "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
        
        return self.model
    
    def grid_search_svm(self, X_train: np.ndarray, y_train: np.ndarray):
        """SVM的超参数搜索"""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC(class_weight='balanced')
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        logger.info("Performing grid search...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), filepath)
        else:
            joblib.dump(self.model, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_type: str = None):
        """加载模型"""
        if model_type == 'cnn':
            from models.eegnet import EEGNet
        
            # 直接从 config 获取参数
            self.model = EEGNet(
                n_channels=self.config.N_CHANNELS,      # 从 config 获取通道数
                n_classes=self.config.N_CLASSES,        # 从 config 获取类别数
                n_timepoints=self.config.N_TIMEPOINTS   # 从 config 获取时间点数
            )
            
            # 加载权重
            state_dict = torch.load(filepath, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()  # 设置为评估模式
            
            logger.info(f"CNN model loaded from {filepath}")
            logger.info(f"Model config: {self.config.N_CHANNELS} channels, {self.config.N_CLASSES} classes")
        else:
            self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")