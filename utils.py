# utils.py
import matplotlib.pyplot as plt
import numpy as np
from mne.viz import plot_topomap
import seaborn as sns

class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_eeg_signals(eeg_data, sfreq=160, channels=None, save_path=None):
        """绘制EEG信号波形"""
        n_channels = min(10, eeg_data.shape[0])
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels))
        
        time = np.arange(eeg_data.shape[1]) / sfreq
        
        for i in range(n_channels):
            axes[i].plot(time, eeg_data[i, :])
            axes[i].set_ylabel(f'Ch{i+1}')
            axes[i].set_xlim([time[0], time[-1]])
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, feature_names=None, save_path=None):
        """绘制特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), 
                      [feature_names[i] if feature_names else f'F{i}' 
                       for i in indices], rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """绘制训练历史曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        
        ax2.plot(history)