# evaluate.py
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
import logging
import torch

logger = logging.getLogger(__name__)

class Evaluator:
    """模型评估器 - 支持 sklearn 和 PyTorch 模型"""
    
    def __init__(self, config=Config):
        self.config = config
        self.results = {}
        
    def _predict_with_model(self, model, X_test: np.ndarray):
        """
        统一的模型预测接口
        支持 sklearn 模型和 PyTorch 模型
        """
        # 判断是否为 PyTorch 模型
        if isinstance(model, torch.nn.Module):
            # PyTorch 模型（CNN）
            model.eval()
            with torch.no_grad():
                # 转换为 tensor
                if isinstance(X_test, np.ndarray):
                    X_test_tensor = torch.FloatTensor(X_test)
                else:
                    X_test_tensor = X_test
                
                # 确保输入形状正确
                if X_test_tensor.dim() == 3:
                    X_test_tensor = X_test_tensor.unsqueeze(1)
                elif X_test_tensor.dim() == 2:
                    # 如果是特征数据，需要 reshape
                    X_test_tensor = X_test_tensor.unsqueeze(1).unsqueeze(-1)
                
                # 预测
                outputs = model(X_test_tensor)
                _, y_pred = torch.max(outputs, 1)
                y_pred = y_pred.numpy()
                
                # 尝试获取概率（softmax）
                try:
                    y_proba = torch.softmax(outputs, dim=1).numpy()
                except:
                    y_proba = None
                    
            return y_pred, y_proba
        else:
            # sklearn 模型（SVM, RF）
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
            else:
                y_pred = model.predict(X_test)
                y_proba = None
            return y_pred, y_proba
    
    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """评估模型性能"""
        # 统一预测接口
        y_pred, y_proba = self._predict_with_model(model, X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        self.results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.results
    
    def plot_confusion_matrix(self, save_path: str = None, 
                             y_true: np.ndarray = None, 
                             y_pred: np.ndarray = None,
                             model=None, X_test=None):
        """
        绘制混淆矩阵
        
        可以多种方式调用：
        1. 先调用 evaluate()，然后直接 plot_confusion_matrix()
        2. 直接传入 y_true, y_pred
        3. 传入 model 和 X_test
        """
        # 获取预测结果
        if y_true is None and 'y_test' in self.results:
            y_true = self.results['y_test']
            y_pred = self.results['y_pred']
        elif y_true is not None and y_pred is None and model is not None and X_test is not None:
            # 从模型预测
            y_pred, _ = self._predict_with_model(model, X_test)
        elif y_true is None or y_pred is None:
            raise ValueError("Need to either run evaluate() first or provide y_true and y_pred")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, n_classes: int = None, save_path: str = None):
        """绘制ROC曲线（多分类）"""
        if 'y_proba' not in self.results or self.results['y_proba'] is None:
            logger.warning("No probability predictions available")
            return
        
        if n_classes is None:
            n_classes = self.results['y_proba'].shape[1]
        
        y_test = self.results['y_test']
        y_proba = self.results['y_proba']
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
                auc = roc_auc_score((y_test == i).astype(int), y_proba[:, i])
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.2f})')
            except:
                logger.warning(f"Could not compute ROC for class {i}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        plt.show()
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """交叉验证（仅支持 sklearn 模型）"""
        # 检查是否为 PyTorch 模型
        if isinstance(model, torch.nn.Module):
            logger.warning("Cross-validation not directly supported for PyTorch models. "
                         "Use sklearn models (SVM/RF) for cross-validation.")
            return None
        
        from sklearn.model_selection import cross_val_score, cross_val_predict
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)
        
        result = {
            'cv_scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'y_pred_cv': y_pred_cv
        }
        
        logger.info(f"Cross-validation scores: {scores}")
        logger.info(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        return result